"""Streamlit entrypoint for Lyric Atlas."""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import time
from collections import Counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv(".env.local", override=True)

from src.analysis import analyze_tokens
from src.categorize import categorize_top_words
from src.dedupe import dedupe_songs
from src.genius_client import (
    ArtistNotFoundError,
    GeniusCloudflareChallengeError,
    GeniusRequestError,
    MissingGeniusTokenError,
    fetch_artist_songs,
    suggest_similar_artists,
)
from src.cache import slugify_artist_name
from src.comparison import normalize_song_dict, shared_top100_rank_table
from src.db import (
    get_archived_analysis_for_artist,
    list_archived_analyses_for_artist,
    list_recent_archives,
    load_analysis_from_supabase,
    save_analysis_to_supabase,
    search_archives_by_artist_name,
    settings_match,
)
from src.lyricsgenius_client import fetch_lyrics_with_diagnostics
from src.r2_store import (
    download_json_from_r2,
    get_corpus_summary,
    get_r2_corpus_key,
    make_empty_corpus,
    r2_object_exists,
    upload_json_to_r2,
)
from src.text_filtering import (
    STOPWORD_VERSION,
    StrictnessMode,
    TokenizationResult,
    aggregate_removed_tokens,
    tokenize_and_filter_lyrics,
)
from src.visualizations import (
    make_category_chart,
    make_comparison_top25_grouped_bar,
    make_top_bigrams_chart,
    make_top_words_chart,
    make_wordcloud_image,
)

SONGS_ANALYZED_METRIC_HELP = (
    "Songs counted in this analysis after dedupe and any manual include/exclude rules. "
    "This is often lower than **Max songs** or your total Genius track list because Genius can include "
    "empty placeholders, unreleased tracks, or songs with no lyric text available."
)

# Set True when a new corpus is ready (or manually hydrated); consumed on first Supabase snapshot for Genius.
_SINGLE_ARTIST_SUPABASE_SAVE_PENDING = "single_artist_supabase_pending_snapshot"


def _streamlit_fragment_fallback(fn):
    return fn


_st_fragment = getattr(st, "fragment", _streamlit_fragment_fallback)


def _flash_notice(message: str, *, icon: str | None = "✅", duration_seconds: int = 5) -> None:
    """Toast notice that auto-dismisses (~5s). Falls back when ``duration`` is unsupported."""
    toast = getattr(st, "toast", None)
    if not callable(toast):
        return
    try:
        toast(message, icon=icon, duration=duration_seconds)
        return
    except TypeError:
        pass
    try:
        toast(message, icon=icon, duration="short")
        return
    except TypeError:
        pass
    try:
        toast(message, icon=icon)
        return
    except TypeError:
        pass
    try:
        toast(message)
    except Exception:
        pass


def _dataframe_single_selected_row_index(widget_state_key: str) -> int | None:
    """Read zero-based row index from ``st.dataframe(..., on_select='rerun', selection_mode='single-row')``."""
    raw = st.session_state.get(widget_state_key)
    if raw is None:
        return None
    if isinstance(raw, dict):
        sel = raw.get("selection") or {}
        rows = sel.get("rows") if isinstance(sel, dict) else None
        if rows:
            return int(rows[0])
        return None
    sel = getattr(raw, "selection", None)
    if sel is not None:
        rows = getattr(sel, "rows", None)
        if rows:
            return int(rows[0])
    return None


@_st_fragment
def _render_genius_lyrics_dismissible_callouts() -> None:
    """Dismiss actions run in a fragment when supported so they do not rerun the full app."""
    if not st.session_state.get("dismiss_genius_mode_info"):
        info_col, dismiss_info_col = st.columns([20, 1])
        with info_col:
            st.info(
                "Genius lyrics mode uses `https://api.genius.com` metadata plus optional lyricsgenius hydration. "
                "It may not include full lyrics; CSV/paste modes are the most reliable for lexical analysis."
            )
        with dismiss_info_col:
            if st.button("x", key="dismiss_genius_mode_info_btn", help="Dismiss this message"):
                st.session_state["dismiss_genius_mode_info"] = True
    if not st.session_state.get("dismiss_lyricsgenius_warning"):
        warning_col, dismiss_warning_col = st.columns([20, 1])
        with warning_col:
            st.warning(
                "lyricsgenius retrieves lyrics by scraping Genius song pages, so it may fail if Genius blocks automated "
                "requests. The official Genius API is still used for metadata."
            )
        with dismiss_warning_col:
            if st.button("x", key="dismiss_lyricsgenius_warning_btn", help="Dismiss this message"):
                st.session_state["dismiss_lyricsgenius_warning"] = True


def _safe_json(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def _lyrics_char_count(value: object) -> int:
    text = str(value or "")
    return len(text) if text else 0


def _sync_lyrics_char_counts_from_lyrics_body(songs: list[dict[str, object]]) -> None:
    """Set lyrics_char_count from lyric text so counts match R2/lyricsgenius merges (avoids stale 0)."""
    for song in songs:
        lyrics = str(song.get("lyrics", "") or "")
        song["lyrics_char_count"] = len(lyrics)


def merge_song_data(
    official_song: dict,
    existing_song: dict | None = None,
    lyricsgenius_result: dict | None = None,
) -> dict:
    """Merge song fields without allowing empty lyrics to overwrite non-empty lyrics."""
    official = dict(official_song or {})
    existing = dict(existing_song or {})
    lg_result = dict(lyricsgenius_result or {})

    official_lyrics = str(official.get("lyrics", "") or "")
    existing_lyrics = str(existing.get("lyrics", "") or "")
    lg_lyrics = str(lg_result.get("lyrics", "") or "")

    print(f"official lyrics char count: {len(official_lyrics)}")
    print(f"existing/corpus lyrics char count: {len(existing_lyrics)}")
    print(f"lyricsgenius lyrics char count: {len(lg_lyrics)}")

    final_lyrics = lg_lyrics or existing_lyrics or official_lyrics or ""

    merged: dict[str, object] = {}
    merged.update(existing)
    merged.update(official)
    merged["lyrics"] = final_lyrics

    if lg_lyrics:
        merged["lyrics_source"] = "lyricsgenius"
    elif existing_lyrics:
        merged["lyrics_source"] = str(existing.get("lyrics_source", "corpus"))
    elif final_lyrics:
        merged["lyrics_source"] = str(merged.get("lyrics_source", "official"))
    else:
        merged["lyrics_source"] = "none"

    if bool(lg_result.get("blocked")):
        merged["lyrics_status"] = "blocked"
    else:
        merged["lyrics_status"] = "available" if final_lyrics else "missing"

    merged["lyrics_char_count"] = len(final_lyrics)
    print(f"final stored lyrics char count: {merged['lyrics_char_count']}")

    if len(lg_lyrics) > 0 and int(merged["lyrics_char_count"]) == 0:
        st.warning("BUG: lyricsgenius returned lyrics, but final song object lost them.")

    return merged


def _song_key(song: dict[str, object]) -> tuple[object, str, str]:
    return (
        song.get("genius_song_id"),
        str(song.get("title", "") or "").strip().lower(),
        str(song.get("artist", "") or "").strip().lower(),
    )


def _song_selection_key(song: dict[str, object]) -> str:
    return "|".join(
        [
            str(song.get("genius_song_id", "") or ""),
            str(song.get("title", "") or "").strip().lower(),
            str(song.get("artist", "") or "").strip().lower(),
        ]
    )


def _song_display_label(song: dict[str, object]) -> str:
    title = str(song.get("title", "") or "").strip() or "(untitled)"
    artist = str(song.get("artist", "") or "").strip() or "(unknown artist)"
    reason = str(song.get("exclude_reason", "") or "").strip()
    if reason:
        return f"{title} - {artist} [{reason}]"
    return f"{title} - {artist}"


def _sort_categories_payload(categories_payload: dict[str, object]) -> dict[str, object]:
    categories = categories_payload.get("categories", [])
    if not isinstance(categories, list):
        return categories_payload
    normalized_categories = []
    for category in categories:
        if not isinstance(category, dict):
            continue
        words = category.get("words", [])
        if not isinstance(words, list):
            words = []
        sorted_words = sorted(
            [word for word in words if isinstance(word, dict)],
            key=lambda item: int(item.get("count", 0) or 0),
            reverse=True,
        )
        total = sum(int(item.get("count", 0) or 0) for item in sorted_words)
        normalized_categories.append(
            {
                "name": category.get("name"),
                "description": category.get("description"),
                "words": sorted_words,
                "_total": total,
            }
        )
    normalized_categories.sort(key=lambda item: int(item.get("_total", 0)), reverse=True)
    for category in normalized_categories:
        category.pop("_total", None)
    return {"categories": normalized_categories}


def _build_analysis_results_payload(
    summary: dict[str, object],
    word_df: pd.DataFrame,
    bigram_df: pd.DataFrame,
    categories_payload: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "summary": summary,
        "songs_analyzed": int(summary.get("total_songs_analyzed", 0)),
        "total_meaningful_tokens": int(summary.get("total_meaningful_tokens", 0)),
        "unique_word_count": int(summary.get("unique_vocabulary_size", 0)),
        "lexical_diversity": float(summary.get("lexical_diversity", 0.0)),
        "top_words": [{"word": str(r["word"]), "count": int(r["count"])} for _, r in word_df.head(100).iterrows()],
        "top_bigrams": [{"bigram": str(r["bigram"]), "count": int(r["count"])} for _, r in bigram_df.head(50).iterrows()],
        "categories": categories_payload,
    }


def _dataframe_from_word_records(records: list[object], word_key: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        key = item.get(word_key)
        if key is None:
            continue
        count_raw = item.get("count", 0)
        try:
            cnt = int(count_raw)
        except (TypeError, ValueError):
            cnt = 0
        rows.append({word_key: str(key), "count": cnt})
    if not rows:
        return pd.DataFrame(columns=[word_key, "count"])
    return pd.DataFrame(rows)


def _render_archived_analysis_block(
    archived_row: dict[str, object],
    *,
    show_wordcloud: bool = True,
    show_filtered_out_diagnostics: bool = False,
    show_lyricsgenius_diagnostics: bool = False,
    widget_namespace: str = "archive",
) -> None:
    row = dict(archived_row)
    summary = dict(row.get("summary_json") or {})
    settings = dict(row.get("settings_json") or {})
    top_words = list(row.get("top_words_json") or [])
    top_bigrams = list(row.get("bigrams_json") or [])
    categories_payload = row.get("categories_json")

    title = str(row.get("artist_name") or "Archived analysis")
    st.markdown(f"## {title}")
    run_id = str(row.get("analysis_run_id") or "")
    run_slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_id)[:64] or "archive"
    plotly_block_id = f"{widget_namespace}_{run_slug}"
    created = str(row.get("created_at") or "")[:19]
    src_used = settings.get("source_used") or "supabase_archive"
    st.caption(f"Imported from Supabase · run `{run_id}` · saved {created} · corpus source: **{src_used}**")
    filt = str(summary.get("filtering_strictness") or row.get("filtering_strictness") or "")
    st.caption(
        f"Filtering: {filt or '—'} (stopword version: {summary.get('stopword_version', STOPWORD_VERSION)}) · "
        f"Dedupe: {row.get('dedupe_mode') or '—'}"
    )

    st.subheader("Analysis Metrics")
    t_songs = int(summary.get("total_songs_analyzed", row.get("songs_analyzed") or 0))
    t_meaningful = int(summary.get("total_meaningful_tokens", row.get("total_meaningful_tokens") or 0))
    u_vocab = int(summary.get("unique_vocabulary_size", row.get("unique_word_count") or 0))
    req_raw = settings.get("requested_song_count")
    try:
        req_n = int(req_raw) if req_raw is not None else None
    except (TypeError, ValueError):
        req_n = None
    t_raw = summary.get("total_tokens_before_filtering")
    if t_raw is not None:
        debug_cols = st.columns(6)
        debug_cols[0].metric("Total songs analyzed", t_songs)
        debug_cols[1].metric("Raw words (pre-cleaning)", int(t_raw))
        debug_cols[2].metric("Meaningful tokens", t_meaningful)
        debug_cols[3].metric("Unique vocabulary", u_vocab)
        debug_cols[4].metric("Lexical diversity", f"{float(summary.get('lexical_diversity', row.get('lexical_diversity') or 0.0)):.3f}")
        debug_cols[5].metric(
            "Requested song count",
            int(req_n) if req_n is not None else t_songs,
            help="From saved settings when present; otherwise matches songs analyzed.",
        )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Songs analyzed", t_songs, help=SONGS_ANALYZED_METRIC_HELP)
    metric_cols[1].metric("Unique words", u_vocab)
    metric_cols[2].metric("Lexical diversity", f"{float(row.get('lexical_diversity') or summary.get('lexical_diversity') or 0.0):.3f}")
    metric_cols[3].metric("Meaningful tokens", t_meaningful)

    word_df = _dataframe_from_word_records(top_words, "word")
    bigram_df = _dataframe_from_word_records(top_bigrams, "bigram")

    if not word_df.empty:
        st.subheader("Top words")
        st.plotly_chart(
            make_top_words_chart(word_df),
            width="stretch",
            key=f"plt_arch_top_words_{plotly_block_id}",
        )
        with st.expander("Top 100 words (copyable text)", expanded=False):
            lines = [
                f"{idx + 1}. {r['word']} ({int(r['count'])})"
                for idx, r in enumerate(word_df.head(100).to_dict("records"))
            ]
            st.text("\n".join(lines))

    if categories_payload:
        st.subheader("Semantic categories")
        chart = make_category_chart(dict(categories_payload))
        if chart is not None:
            st.plotly_chart(
                chart,
                width="stretch",
                key=f"plt_arch_categories_{plotly_block_id}",
            )
        with st.expander("Raw categories JSON", expanded=False):
            st.json(categories_payload)

    if not bigram_df.empty:
        st.subheader("Top bigrams")
        st.plotly_chart(
            make_top_bigrams_chart(bigram_df),
            width="stretch",
            key=f"plt_arch_bigrams_{plotly_block_id}",
        )

    if show_filtered_out_diagnostics:
        st.subheader("Filtered-out words diagnostics")
        if "total_tokens_before_filtering" in summary:
            diag_cols = st.columns(5)
            diag_cols[0].metric("Total raw tokens", int(summary["total_tokens_before_filtering"]))
            diag_cols[1].metric(
                "Removed by standard stopwords",
                int(summary.get("removed_standard_stopwords", 0)),
            )
            diag_cols[2].metric(
                "Removed by definite lyric stopwords",
                int(summary.get("removed_definite_lyric_stopwords", 0)),
            )
            diag_cols[3].metric(
                "Removed by maybe lyric stopwords",
                int(summary.get("removed_maybe_lyric_stopwords", 0)),
            )
            diag_cols[4].metric("Meaningful tokens remaining", int(summary.get("total_meaningful_tokens", t_meaningful)))
            st.info(
                "The per-token **removed-word table** is not stored in Supabase archives — "
                "only aggregate removal counts above."
            )
        else:
            st.info("This archive predates detailed filtering stats, or none were saved. Run a fresh analysis for full diagnostics.")

    if show_lyricsgenius_diagnostics:
        st.subheader("lyricsgenius diagnostics")
        st.info(
            "Hydration diagnostics are not stored on the archive. "
            "They reflect the current session only after you run **Single artist** hydration."
        )

    if show_wordcloud:
        st.subheader("Word cloud")
        if word_df.empty:
            st.info("No word frequencies available for the cloud.")
        else:
            cap_hi = min(200, max(len(word_df), 5))
            cap_lo = min(5, cap_hi)
            wc_top_n = st.slider(
                "Top words to include in the cloud",
                min_value=cap_lo,
                max_value=cap_hi,
                value=min(25, cap_hi),
                step=5,
                key=f"plt_arch_wc_top_{plotly_block_id}",
                help="Uses frequencies stored on this archive (same ranking as Top words).",
            )
            image = make_wordcloud_image(word_df, top_n=int(wc_top_n))
            if image is not None:
                st.image(image, width="stretch")

    with st.expander("Included / excluded songs", expanded=False):
        st.info(
            "Per-song included and excluded lists are **not** stored in Supabase archives. "
            "Open the linked R2 corpus or run a **fresh analysis** for song-level tables."
        )
        rk = str(row.get("r2_corpus_key") or "").strip()
        if rk:
            st.code(rk, language="text")

    filt_for_export = str(summary.get("filtering_strictness") or filt or "")
    sw_ver = str(summary.get("stopword_version", STOPWORD_VERSION))

    st.subheader("Exports")
    export_cols = st.columns(3)
    words_export_df = word_df.copy()
    words_export_df["filtering_strictness"] = filt_for_export
    words_export_df["stopword_version"] = sw_ver
    bigrams_export_df = bigram_df.copy()
    bigrams_export_df["filtering_strictness"] = filt_for_export
    bigrams_export_df["stopword_version"] = sw_ver
    analysis_settings = {
        "import_source": "supabase_archive",
        "analysis_run_id": run_id,
        "saved_settings": settings,
        "dedupe_mode": row.get("dedupe_mode"),
        "filtering_strictness": filt_for_export,
        "stopword_version": sw_ver,
    }
    safe_artist = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:80] or "analysis"
    export_cols[0].download_button(
        "Download words CSV",
        data=words_export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{safe_artist}_word_frequencies.csv",
        mime="text/csv",
    )
    export_cols[1].download_button(
        "Download bigrams CSV",
        data=bigrams_export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{safe_artist}_bigram_frequencies.csv",
        mime="text/csv",
    )
    export_cols[2].download_button(
        "Download analysis JSON",
        data=_safe_json(
            {
                "summary": summary,
                "top_words": top_words,
                "top_bigrams": top_bigrams,
                "categories": categories_payload,
                "analysis_settings": analysis_settings,
            }
        ),
        file_name=f"{safe_artist}_lyric_atlas_analysis.json",
        mime="application/json",
    )


def _render_archive_explorer(artist_metadata: dict[str, object]) -> None:
    """Diagnostics panel: list archived runs and allow loading any run by id."""
    genius_artist_id = str(artist_metadata.get("artist_id") or artist_metadata.get("genius_artist_id") or "")
    if not genius_artist_id:
        st.info("Archive explorer requires artist metadata with a Genius artist ID.")
        return

    st.subheader("Archive Explorer")
    runs = list_archived_analyses_for_artist(genius_artist_id)
    if not runs:
        st.info("No archived analyses found for this artist in Supabase.")
        return

    run_ids = [str(r.get("analysis_run_id", "")) for r in runs if str(r.get("analysis_run_id", ""))]
    if not run_ids:
        return

    explorer_sig = f"{genius_artist_id}|{','.join(run_ids)}"
    if st.session_state.get("_archive_explorer_table_sig") != explorer_sig:
        st.session_state["_archive_explorer_table_sig"] = explorer_sig
        st.session_state["_archive_explorer_table_nonce"] = int(st.session_state.get("_archive_explorer_table_nonce", 0)) + 1
    table_key = f"archive_explorer_sel_{st.session_state.get('_archive_explorer_table_nonce', 0)}"

    preview_df = _archives_browse_preview_rows(runs)
    st.caption("**Select one row** in the table, then click **Load selected archive**.")
    st.dataframe(
        preview_df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
    )

    idx = _dataframe_single_selected_row_index(table_key)
    if idx is None:
        st.info("Select a row in the table above, then load.")
    elif idx < 0 or idx >= len(runs):
        st.warning("Selection is out of date; try selecting the row again.")
        idx = None

    if st.button(
        "Load selected archive",
        key="archive_explorer_load_btn",
        disabled=idx is None,
    ):
        target = runs[idx] if idx is not None else None
        selected_run_id = str((target or {}).get("analysis_run_id") or "")
        row = load_analysis_from_supabase(selected_run_id) if selected_run_id else None
        if not row:
            st.warning("Selected archive run was not found.")
            return
        st.session_state["archived_analysis_row"] = row
        st.session_state["archive_decision"] = "load"
        st.session_state["workflow_stage"] = "archive_loaded"
        _flash_notice(f"Loaded archive {selected_run_id}.", icon="📂")


def _merge_candidate_with_corpus(
    candidate_songs: list[dict[str, object]],
    corpus_songs: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_key = {_song_key(song): dict(song) for song in corpus_songs}
    merged: list[dict[str, object]] = []
    for candidate in candidate_songs:
        existing = by_key.get(_song_key(candidate), {})
        final_lyrics = (
            str(existing.get("lyrics", "") or "")
            or str(candidate.get("lyrics", "") or "")
            or ""
        )
        row = dict(candidate)
        row["lyrics"] = final_lyrics
        row["lyrics_char_count"] = len(final_lyrics)
        row["lyrics_status"] = "available" if final_lyrics else "missing"
        row["lyrics_source"] = str(existing.get("lyrics_source", row.get("lyrics_source", "none")))
        merged.append(row)
    return merged


def update_song_lyrics_in_corpus(
    corpus_songs: list[dict[str, object]],
    song_key: dict[str, object],
    lyrics: str,
    source: str,
    blocked: bool = False,
) -> None:
    """Update a matching song in corpus without overwriting non-empty lyrics with empty text."""
    target_id = song_key.get("genius_song_id")
    target_title = str(song_key.get("title", "") or "").strip().lower()
    target_artist = str(song_key.get("artist", "") or "").strip().lower()
    for song in corpus_songs:
        same_id = target_id is not None and song.get("genius_song_id") == target_id
        same_text = (
            str(song.get("title", "") or "").strip().lower() == target_title
            and str(song.get("artist", "") or "").strip().lower() == target_artist
        )
        if not (same_id or same_text):
            continue
        existing = str(song.get("lyrics", "") or "")
        new_lyrics = str(lyrics or "")
        if new_lyrics:
            song["lyrics"] = new_lyrics
            song["lyrics_source"] = source
            song["lyrics_status"] = "available"
            song["lyrics_char_count"] = len(new_lyrics)
        elif blocked:
            song["lyrics_status"] = "blocked"
            song["lyrics_source"] = "none"
            song["lyrics_char_count"] = len(existing)
        elif not existing:
            song["lyrics_source"] = "none"
            song["lyrics_status"] = "missing"
            song["lyrics_char_count"] = 0
        return


def _run_analysis_pipeline(
    raw_songs: list[dict[str, object]],
    dedupe_mode: str,
    exclude_versions: bool,
    filtering_strictness: StrictnessMode,
    manually_included_song_keys: set[str] | None = None,
    manually_excluded_song_keys: set[str] | None = None,
    canonical_artist_name: str | None = None,
):
    dedupe_result = dedupe_songs(
        songs=raw_songs,
        mode=dedupe_mode,
        exclude_versions=exclude_versions,
        canonical_artist_name=canonical_artist_name,
    )
    included_songs = dedupe_result.included
    excluded_songs = dedupe_result.excluded
    if manually_included_song_keys:
        manually_reincluded = [song for song in excluded_songs if _song_selection_key(song) in manually_included_song_keys]
        if manually_reincluded:
            included_songs = included_songs + manually_reincluded
            excluded_songs = [song for song in excluded_songs if _song_selection_key(song) not in manually_included_song_keys]
    if manually_excluded_song_keys:
        manually_removed = [song for song in included_songs if _song_selection_key(song) in manually_excluded_song_keys]
        if manually_removed:
            excluded_songs = excluded_songs + [{**song, "exclude_reason": "Manually excluded from included songs"} for song in manually_removed]
            included_songs = [song for song in included_songs if _song_selection_key(song) not in manually_excluded_song_keys]
    if not included_songs:
        return None, included_songs, excluded_songs, {}

    tokenization_results: list[TokenizationResult] = [
        tokenize_and_filter_lyrics(str(song.get("lyrics", "") or ""), strictness=filtering_strictness) for song in included_songs
    ]
    per_song_tokens = [result.meaningful_tokens for result in tokenization_results]
    prefilter_counts = [len(result.raw_tokens) for result in tokenization_results]
    analysis_result = analyze_tokens(included_songs, per_song_tokens, prefilter_counts)

    aggregate_counts = Counter()
    for result in tokenization_results:
        aggregate_counts.update(result.counts_by_reason)

    analysis_result.summary["filtering_strictness"] = filtering_strictness
    analysis_result.summary["stopword_version"] = STOPWORD_VERSION
    analysis_result.summary["removed_standard_stopwords"] = int(aggregate_counts.get("standard_stopword", 0))
    analysis_result.summary["removed_definite_lyric_stopwords"] = int(aggregate_counts.get("definite_lyric_stopword", 0))
    analysis_result.summary["removed_maybe_lyric_stopwords"] = int(aggregate_counts.get("maybe_lyric_stopword", 0))
    analysis_result.summary["removed_too_short"] = int(aggregate_counts.get("too_short", 0))
    analysis_result.summary["removed_non_alpha"] = int(aggregate_counts.get("non_alpha", 0))

    diagnostics = {
        "tokenization_results": tokenization_results,
        "removed_words_top_50": aggregate_removed_tokens(tokenization_results),
        "counts_by_reason": dict(aggregate_counts),
    }
    return analysis_result, included_songs, excluded_songs, diagnostics


def _songs_from_uploaded_csv(uploaded_file) -> list[dict[str, object]]:
    df = pd.read_csv(uploaded_file)
    expected = {"title", "artist", "lyrics"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {', '.join(sorted(missing))}")

    songs: list[dict[str, object]] = []
    for _, row in df.iterrows():
        songs.append(
            {
                "title": str(row.get("title", "")),
                "artist": str(row.get("artist", "")),
                "genius_song_id": None,
                "url": row.get("url"),
                "album": row.get("album"),
                "release_date": row.get("release_date"),
                "lyrics": str(row.get("lyrics", "")),
                "lyrics_source": "csv",
                "lyrics_status": "available" if str(row.get("lyrics", "") or "").strip() else "missing",
                "lyrics_char_count": len(str(row.get("lyrics", "") or "")),
            }
        )
    return songs


def _render_paste_mode_inputs() -> None:
    if "pasted_songs" not in st.session_state:
        st.session_state["pasted_songs"] = []

    st.subheader("Paste lyrics")
    with st.form("paste_lyrics_form"):
        title = st.text_input("Title")
        artist = st.text_input("Artist")
        album = st.text_input("Album (optional)")
        lyrics = st.text_area("Lyrics")
        add_song = st.form_submit_button("Add pasted song")

    if add_song:
        if not title.strip() or not artist.strip() or not lyrics.strip():
            st.error("Title, artist, and lyrics are required to add a pasted song.")
        else:
            st.session_state["pasted_songs"].append(
                {
                    "title": title.strip(),
                    "artist": artist.strip(),
                    "genius_song_id": None,
                    "url": None,
                    "album": album.strip() or None,
                    "release_date": None,
                    "lyrics": lyrics,
                    "lyrics_source": "paste",
                    "lyrics_status": "available" if lyrics.strip() else "missing",
                    "lyrics_char_count": len(lyrics or ""),
                }
            )
            st.success(f"Added pasted song: {title.strip()}")

    if st.session_state["pasted_songs"]:
        st.caption(f"Pasted songs in memory: {len(st.session_state['pasted_songs'])}")
        st.dataframe(
            pd.DataFrame(st.session_state["pasted_songs"])[["title", "artist", "album"]],
            width="stretch",
        )
        if st.button("Clear pasted songs"):
            st.session_state["pasted_songs"] = []
            st.rerun()


def _collect_raw_songs(
    input_mode: str,
    artist_name: str,
    max_songs: int,
    include_features: bool,
    uploaded_csv,
    auto_hydrate_workers: int = 4,
) -> tuple[list[dict[str, object]], str]:
    if input_mode == "CSV upload mode":
        if uploaded_csv is None:
            raise ValueError("Please upload a CSV with columns: title, artist, lyrics.")
        return _songs_from_uploaded_csv(uploaded_csv), "CSV upload mode"

    if input_mode == "Paste lyrics mode":
        songs = st.session_state.get("pasted_songs", [])
        if not songs:
            raise ValueError("Add at least one pasted song before running analysis.")
        return songs, "Paste lyrics mode"

    if not artist_name.strip():
        raise ValueError("Please enter an artist name for official Genius metadata mode.")
    if not os.getenv("GENIUS_ACCESS_TOKEN"):
        raise MissingGeniusTokenError("Missing GENIUS_ACCESS_TOKEN. Add it to `.env.local` and restart the app.")

    payload = fetch_artist_songs(
        artist_name=artist_name,
        max_songs=max_songs,
        include_features=include_features,
    )
    if not payload.get("songs"):
        typed = artist_name.strip()
        suggestions = suggest_similar_artists(typed) if typed else []
        raise ArtistNotFoundError(typed, suggestions)
    st.session_state["official_api_exact_results"] = payload.get("songs", [])
    existing_lookup = {_song_key(song): song for song in st.session_state.get("corpus_songs", [])}
    songs: list[dict[str, object]] = []
    returned_songs = list(payload.get("songs", []))[:max_songs]
    for official_song in returned_songs:
        official_base = dict(official_song)
        official_base["lyrics"] = str(official_base.get("lyrics", "") or "")
        official_base["lyrics_status"] = "available" if official_base["lyrics"] else "missing"
        official_base["lyrics_source"] = "official_api" if official_base["lyrics"] else "none"
        official_base["lyrics_char_count"] = len(official_base["lyrics"])
        official_base["official_api_lyrics_char_count"] = len(str(official_song.get("lyrics", "") or ""))
        merged_song = merge_song_data(
            official_song=official_base,
            existing_song=existing_lookup.get(_song_key(official_base)),
            lyricsgenius_result=None,
        )
        songs.append(merged_song)

    auto_hydrate_targets = {
        _song_key(song) for song in songs if int(song.get("lyrics_char_count", 0) or 0) == 0
    }

    if auto_hydrate_targets:
        st.info(
            f"Auto-hydrating {len(auto_hydrate_targets)} song(s) missing lyrics."
        )
        try:
            songs, diagnostics_rows = hydrate_missing_lyrics_with_lyricsgenius(
                corpus_songs=songs,
                max_workers=max(1, int(auto_hydrate_workers)),
                target_song_keys=auto_hydrate_targets,
            )
            st.session_state["lyricsgenius_hydration_diagnostics"] = diagnostics_rows
        except Exception as exc:  # noqa: BLE001
            st.warning(
                f"Auto-hydration interrupted ({exc!r}). "
                "Lyrics updated before the error are kept; hydrate again later for any still missing."
            )

    songs = songs[:max_songs]
    return songs, "Official Genius metadata mode"


def hydrate_missing_lyrics_with_lyricsgenius(
    corpus_songs: list[dict],
    max_workers: int = 2,
    target_song_keys: set[tuple[object, str, str]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Hydrate missing lyrics in-place using controlled parallel lyricsgenius calls."""
    diagnostics_rows: list[dict[str, object]] = []
    progress = st.progress(0.0)
    progress_text = st.empty()
    missing_candidates = [song for song in corpus_songs if int(song.get("lyrics_char_count", 0) or 0) == 0]
    if target_song_keys:
        missing_candidates = [song for song in missing_candidates if _song_key(song) in target_song_keys]
    if not missing_candidates:
        st.info("No missing lyrics to hydrate.")
        st.session_state["lyricsgenius_hydration_diagnostics"] = diagnostics_rows
        st.session_state["hydration_fetch_status"] = "No songs needed hydration."
        return corpus_songs, diagnostics_rows

    total_missing = len(missing_candidates)
    st.session_state["hydration_fetch_status"] = f"Fetching lyrics: 0 / {total_missing} songs"
    progress_text.markdown(f"**Fetching lyrics:** 0 / {total_missing} songs")

    fetched_count = 0
    missing_count = 0
    blocked_count = 0
    error_count = 0
    processed = 0
    saw_blocked = False
    batch_size = max_workers * 2

    def _worker(song: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        time.sleep(random.uniform(0.5, 2.0))
        diagnostics = fetch_lyrics_with_diagnostics(
            title=str(song.get("title", "") or ""),
            artist=str(song.get("artist", "") or ""),
        )
        return song, diagnostics

    for start_idx in range(0, len(missing_candidates), batch_size):
        if saw_blocked and start_idx > 0:
            cooldown = min(15.0, 4.0 + (start_idx / max(len(missing_candidates), 1)) * 8.0)
            time.sleep(cooldown)
        batch = missing_candidates[start_idx : start_idx + batch_size]

        if max_workers == 1:
            completed: list[tuple[dict[str, object], dict[str, object]]] = []
            for song in batch:
                try:
                    completed.append(_worker(song))
                except Exception as exc:  # noqa: BLE001
                    completed.append(
                        (
                            song,
                            {
                                "input_title": song.get("title"),
                                "input_artist": song.get("artist"),
                                "returned_title": None,
                                "returned_artist": None,
                                "found_song": False,
                                "lyrics": "",
                                "lyrics_char_count": 0,
                                "blocked": False,
                                "error_message": str(exc),
                            },
                        )
                    )
        else:
            completed = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_worker, song) for song in batch]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        completed.append(future.result())
                    except Exception as exc:  # noqa: BLE001
                        completed.append(
                            (
                                {"title": "unknown", "artist": "unknown"},
                                {
                                    "input_title": "unknown",
                                    "input_artist": "unknown",
                                    "returned_title": None,
                                    "returned_artist": None,
                                    "found_song": False,
                                    "lyrics": "",
                                    "lyrics_char_count": 0,
                                    "blocked": False,
                                    "error_message": str(exc),
                                },
                            )
                        )

        for song, diagnostics in completed:
            processed += 1
            lyrics_text = str(diagnostics.get("lyrics", "") or "")
            blocked = bool(diagnostics.get("blocked"))
            has_lyrics = bool(lyrics_text.strip())

            if has_lyrics:
                update_song_lyrics_in_corpus(corpus_songs, song, lyrics_text, "lyricsgenius")
                fetched_count += 1
            elif blocked:
                update_song_lyrics_in_corpus(corpus_songs, song, "", "none", blocked=True)
                blocked_count += 1
                saw_blocked = True
            else:
                update_song_lyrics_in_corpus(corpus_songs, song, "", "none")
                missing_count += 1

            if diagnostics.get("error_message"):
                error_count += 1

            updated_song = next((s for s in corpus_songs if _song_key(s) == _song_key(song)), song)
            diagnostics_rows.append(
                {
                    "input_title": diagnostics.get("input_title"),
                    "input_artist": diagnostics.get("input_artist"),
                    "returned_title": diagnostics.get("returned_title"),
                    "returned_artist": diagnostics.get("returned_artist"),
                    "found_song": diagnostics.get("found_song"),
                    "lyricsgenius_lyrics_char_count": diagnostics.get("lyrics_char_count"),
                    "blocked": blocked,
                    "error_message": diagnostics.get("error_message"),
                    "final_stored_lyrics_char_count": updated_song.get("lyrics_char_count", 0),
                }
            )

            progress.progress(min(processed / total_missing, 1.0))
            status_line = f"Fetching lyrics: {processed} / {total_missing} songs"
            progress_text.markdown(f"**{status_line}**")
            st.session_state["hydration_fetch_status"] = status_line

    if saw_blocked:
        final_msg = (
            f"Last hydration: finished {processed} / {total_missing} attempts. "
            "At least one response was blocked or empty; any lyrics retrieved are kept."
        )
        st.session_state["hydration_fetch_status"] = final_msg
        progress.progress(1.0)
        st.warning(
            "Genius may be blocking or rate-limiting lyric fetches. Partial lyrics are still saved per song — "
            "use **Run analysis** to see them, or hydrate again after a pause (fewer parallel workers may help)."
        )
    else:
        final_msg = f"Last hydration: completed {processed} / {total_missing} songs."
        st.session_state["hydration_fetch_status"] = final_msg
        progress.progress(1.0)
    progress_text.markdown(f"**{final_msg}**")

    st.session_state["lyricsgenius_hydration_diagnostics"] = diagnostics_rows

    if any(int(row.get("lyricsgenius_lyrics_char_count", 0) or 0) > 0 for row in diagnostics_rows):
        if not any(int(song.get("lyrics_char_count", 0) or 0) > 0 for song in corpus_songs):
            st.warning("BUG: lyricsgenius returned lyrics, but the hydrated lyrics were not written into corpus_songs.")
    st.info(
        f"Hydration summary: fetched_count={fetched_count}, missing_count={missing_count}, "
        f"blocked_count={blocked_count}, error_count={error_count}"
    )
    return corpus_songs, diagnostics_rows


def _best_matching_archive_row_for_compare(genius_artist_id: str, settings: dict[str, object]) -> dict[str, object] | None:
    """Archive row only if settings match — no fallback to another artist's archive."""
    if not str(genius_artist_id or "").strip():
        return None
    rows = list_archived_analyses_for_artist(genius_artist_id)
    matching = [r for r in rows if settings_match(r, settings)]
    if not matching:
        return None
    return max(
        matching,
        key=lambda r: (int(r.get("songs_analyzed") or 0), str(r.get("created_at") or "")),
    )


def _load_comparison_artist_corpus(
    artist_query: str,
    max_songs: int,
    include_features: bool,
    *,
    archive_settings: dict[str, object],
    parallel_workers: int = 2,
    status_log=None,
) -> tuple[list[dict[str, object]], str, str]:
    """Load songs: Genius metadata + merge with R2; prefer Supabase-linked r2_corpus_key when settings match."""
    aq = artist_query.strip()
    payload = fetch_artist_songs(
        aq,
        max_songs=max_songs,
        include_features=include_features,
    )
    if not payload.get("songs"):
        suggestions = suggest_similar_artists(aq)
        raise ArtistNotFoundError(aq, suggestions)

    label = str(payload.get("artist_name") or aq)
    artist_meta: dict[str, object] = {
        "artist_name": payload.get("artist_name") or aq,
        "artist_id": payload.get("artist_id"),
        "slug": slugify_artist_name(str(payload.get("artist_name") or aq)),
    }
    genius_id = str(artist_meta.get("artist_id") or "")

    candidate_raw = list(payload.get("songs", []))[:max_songs]
    candidates = [normalize_song_dict(s) for s in candidate_raw]

    archive = None
    if genius_id:
        try:
            archive = _best_matching_archive_row_for_compare(genius_id, archive_settings)
        except Exception:
            archive = None
            if status_log:
                status_log(f"**{label}**: Supabase archive lookup failed — using Genius + default R2 key only.")
    r2_key = ""
    if archive and str(archive.get("r2_corpus_key") or "").strip():
        r2_key = str(archive["r2_corpus_key"]).strip()
        if status_log:
            status_log(f"**{label}**: Supabase archive matched → loading corpus from linked R2 key.")
    if not r2_key:
        r2_key = get_r2_corpus_key(artist_meta)
        if status_log:
            status_log(f"**{label}**: using default R2 key `{r2_key}`.")

    r2_doc: dict[str, object] | None = None
    if r2_object_exists(r2_key):
        r2_doc = download_json_from_r2(r2_key)
    if not r2_doc:
        r2_doc = make_empty_corpus(artist_meta)
        if status_log:
            status_log(f"**{label}**: no object in R2 yet (empty corpus shell).")

    r2_songs = list(r2_doc.get("songs", [])) if isinstance(r2_doc.get("songs", []), list) else []
    merged = _merge_candidate_with_corpus(candidates, r2_songs)
    merged_cap = merged[:max_songs]

    with_lyrics_count = sum(1 for s in merged_cap if int(s.get("lyrics_char_count", 0) or 0) > 0)
    did_hydrate = False
    if with_lyrics_count < max_songs:
        missing_target_keys = {
            _song_key(s) for s in merged_cap if int(s.get("lyrics_char_count", 0) or 0) == 0
        }
        if missing_target_keys:
            if status_log:
                status_log(f"**{label}**: hydrating {len(missing_target_keys)} song(s) via lyricsgenius…")
            try:
                merged_cap, _ = hydrate_missing_lyrics_with_lyricsgenius(
                    merged_cap,
                    max_workers=max(1, int(parallel_workers)),
                    target_song_keys=missing_target_keys,
                )
            except Exception as exc:  # noqa: BLE001
                if status_log:
                    status_log(
                        f"**{label}**: hydration error ({exc!r}); continuing with any lyrics fetched so far."
                    )
            did_hydrate = True
        merged_cap = merged_cap[:max_songs]

    if did_hydrate:
        out_doc = dict(r2_doc)
        out_doc["songs"] = merged_cap
        try:
            upload_json_to_r2(r2_key, out_doc)
            summary = get_corpus_summary(out_doc)
            _flash_notice(
                f"R2 corpus updated for {label} ({summary['songs_with_lyrics']}/{summary['song_count']} songs with lyrics)."
            )
            if status_log:
                status_log(
                    f"**{label}**: wrote R2 corpus (songs_with_lyrics={summary['songs_with_lyrics']}/{summary['song_count']})."
                )
        except Exception as exc:  # noqa: BLE001
            if status_log:
                status_log(f"**{label}**: R2 upload failed ({exc}); continuing with in-memory corpus.")

    note_parts: list[str] = ["Genius metadata"]
    if archive:
        note_parts.append("Supabase archive")
    note_parts.append("R2")
    if did_hydrate:
        note_parts.append("lyricsgenius")
    return merged_cap, label, " + ".join(note_parts)


def _render_compare_artists_tab() -> None:
    """Compare lexical diversity and word rankings across 2–5 artists."""
    st.subheader("Compare artists")
    st.caption(
        "Uses the same token filtering as **Single artist**. For each name we fetch Genius metadata, "
        "merge with the **R2** lyric corpus (same keys as Single artist), and when a **Supabase** archive exists "
        "with the same deduping / filtering / featured settings below, we use its linked `r2_corpus_key` when one exists "
        "(archives de-duplicate by those settings; larger song counts replace smaller ones). "
        "Missing lyrics are hydrated with lyricsgenius and the corpus is written back to R2."
    )
    if not os.getenv("GENIUS_ACCESS_TOKEN"):
        st.error("Add `GENIUS_ACCESS_TOKEN` to `.env.local` to load artist metadata from Genius.")
        return

    strictness_map_cmp: dict[str, StrictnessMode] = {
        "Basic": "basic",
        "Lyric-clean": "lyric_clean",
        "Theme-focused": "theme_focused",
    }

    raw_lines = st.text_area(
        "Artist names (2–5, one per line)",
        height=120,
        placeholder="Noah Kahan\nTaylor Swift",
        key="compare_artist_lines",
    )
    max_cmp = int(
        st.number_input(
            "Max songs per artist",
            min_value=10,
            max_value=500,
            value=50,
            step=5,
            key="cmp_max_songs",
        )
    )
    include_feat_cmp = st.checkbox("Include featured songs", value=False, key="cmp_include_features")
    parallel_cmp_workers = int(
        st.slider(
            "Parallel lyric fetch workers (compare)",
            min_value=1,
            max_value=50,
            value=2,
            help="Used when lyricsgenius must hydrate missing lyrics for an artist.",
            key="cmp_parallel_lyric_workers",
        )
    )
    dedupe_cmp = st.radio(
        "Deduping mode",
        [
            "Strict canonical songs only",
            "Keep major alternate versions",
            "Keep everything",
        ],
        index=0,
        key="cmp_dedupe",
    )
    strict_label_cmp = st.selectbox(
        "Filtering strictness",
        options=["Basic", "Lyric-clean", "Theme-focused"],
        index=1,
        key="cmp_filtering_strictness",
    )
    exclude_versions_cmp = dedupe_cmp == "Strict canonical songs only"
    filtering_cmp = strictness_map_cmp[strict_label_cmp]
    archive_settings_cmp: dict[str, object] = {
        "filtering_strictness": filtering_cmp,
        "dedupe_mode": dedupe_cmp,
        "include_features": include_feat_cmp,
    }

    artists_input = [ln.strip() for ln in (raw_lines or "").splitlines() if ln.strip()]
    if len(artists_input) > 5:
        artists_input = artists_input[:5]
        st.warning("Using only the first five artist names.")

    if st.button("Run comparison", type="primary", key="cmp_run_button"):
        if len(artists_input) < 2:
            st.error("Enter at least two artist names (one per line).")
            return

        results: list[tuple[str, object]] = []
        word_dfs: dict[str, pd.DataFrame] = {}

        with st.status("Loading artists and analyzing…", expanded=True) as status_cmp:
            for i, query in enumerate(artists_input):
                try:
                    corpus, label, note = _load_comparison_artist_corpus(
                        query,
                        max_cmp,
                        include_feat_cmp,
                        archive_settings=archive_settings_cmp,
                        parallel_workers=parallel_cmp_workers,
                        status_log=status_cmp.write,
                    )
                except ArtistNotFoundError as exc:
                    st.error(
                        f"No songs for **{exc.typed_name!r}**. "
                        "Check spelling or try the Single-artist tab to see Genius suggestions."
                    )
                    if exc.suggestions:
                        st.caption("Did you mean: " + ", ".join(exc.suggestions))
                    return
                except (GeniusRequestError, GeniusCloudflareChallengeError, MissingGeniusTokenError) as exc:
                    st.error(str(exc))
                    return

                songs_with_lyrics = [s for s in corpus if int(s.get("lyrics_char_count", 0) or 0) > 0]
                if not songs_with_lyrics:
                    st.warning(
                        f"**{label}** has no lyric text after R2 + hydration ({note}). "
                        "Run **Single artist** once for this artist to build the R2 corpus, or check Genius/lyricsgenius availability."
                    )
                    return

                status_cmp.write(f"Analyzing **{label}** ({len(songs_with_lyrics)} songs with lyrics, {note})…")
                analysis_cmp, _, _, _ = _run_analysis_pipeline(
                    raw_songs=corpus,
                    dedupe_mode=dedupe_cmp,
                    exclude_versions=exclude_versions_cmp,
                    filtering_strictness=filtering_cmp,
                    manually_included_song_keys=None,
                    manually_excluded_song_keys=None,
                    canonical_artist_name=query,
                )
                if analysis_cmp is None:
                    st.warning(f"No songs remained after filters for **{label}**.")
                    return
                results.append((label, analysis_cmp))
                word_dfs[label] = analysis_cmp.word_frequencies_df.copy()

        st.markdown("### Lexical diversity")
        div_cols = st.columns(len(results))
        for idx, (label, ar) in enumerate(results):
            with div_cols[idx]:
                st.metric(
                    label,
                    f"{float(ar.summary['lexical_diversity']):.3f}",
                    help="Unique meaningful tokens ÷ total meaningful tokens",
                )

        st.markdown("### Top 25 words (grouped)")
        fig_cmp = make_comparison_top25_grouped_bar(word_dfs)
        if fig_cmp is not None:
            st.plotly_chart(fig_cmp, width="stretch", key="plt_compare_top25_grouped")

        st.markdown("### Shared words in each artist’s top 100 (relative ranks)")
        shared_df = shared_top100_rank_table(word_dfs)
        if shared_df.empty:
            st.info("No single word appears in every artist’s top 100 at the same time. Try fewer artists or overlapping vocabularies.")
        else:
            st.caption(
                f"{len(shared_df)} words appear in all {len(word_dfs)} top-100 lists. "
                "Rank 1 = most frequent for that artist."
            )
            st.dataframe(shared_df, width="stretch")


def _clear_single_artist_corpus_session() -> None:
    """Drop loaded corpus / workflow so a new sidebar query cannot show the previous artist."""
    for k in (
        "artist_metadata",
        "candidate_songs",
        "r2_corpus",
        "r2_corpus_key",
        "archived_analysis_row",
        "archive_decision",
        "corpus_display_title",
        "corpus_artist_name",
        "corpus_max_songs",
        "corpus_include_features",
        "corpus_input_mode",
        "corpus_source_used",
        "official_api_exact_results",
        "lyricsgenius_hydration_diagnostics",
        "hydration_fetch_status",
        _SINGLE_ARTIST_SUPABASE_SAVE_PENDING,
    ):
        st.session_state.pop(k, None)
    st.session_state["workflow_stage"] = "idle"
    st.session_state["corpus_songs"] = []


_ACTIVE_CORPUS_STAGES = frozenset(
    {
        "metadata_loaded",
        "archive_found",
        "archive_loaded",
        "checking_r2",
        "r2_loaded",
        "hydrating_lyrics",
        "ready_to_analyze",
        "analysis_complete",
    }
)


def _sidebar_drifted_from_loaded_corpus(
    *,
    input_mode: str,
    artist_name: str,
    max_songs: int,
    include_features: bool,
) -> bool:
    """True when sidebar no longer matches the load that produced the current workflow / corpus."""
    stage = st.session_state.get("workflow_stage", "idle")
    if stage not in _ACTIVE_CORPUS_STAGES:
        return False
    stored_mode = st.session_state.get("corpus_input_mode")
    if stored_mode is not None and input_mode != stored_mode:
        return True
    if int(max_songs) != int(st.session_state.get("corpus_max_songs", -1)):
        return True
    if input_mode == "Genius lyrics":
        if bool(include_features) != bool(st.session_state.get("corpus_include_features", False)):
            return True
        cur_slug = slugify_artist_name(artist_name)
        prev_slug = slugify_artist_name(str(st.session_state.get("corpus_artist_name", "")))
        if cur_slug and prev_slug and cur_slug != prev_slug:
            return True
    return False


def _render_clickable_artist_suggestions(
    suggestions: list[str],
    *,
    session_key_pending: str,
    input_session_key: str,
    widget_key_prefix: str,
) -> None:
    """Show Genius search corrections; picking one fills the artist field and re-runs with sidebar settings."""
    if not suggestions:
        return
    st.caption("Use your current sidebar settings — pick a correction to run again:")
    n_cols = min(3, len(suggestions))
    cols = st.columns(n_cols)
    for idx, name in enumerate(suggestions):
        slug = slugify_artist_name(name)[:40] or "artist"
        with cols[idx % n_cols]:
            if st.button(name, key=f"{widget_key_prefix}_{idx}_{slug}"):
                st.session_state[input_session_key] = name
                st.session_state[session_key_pending] = True


def _render_single_artist_tab() -> None:
    """Single-artist Streamlit workflow (sidebar + analysis)."""
    with st.sidebar:
        st.header("Controls")
        settings_tab, diagnostics_tab = st.tabs(["Settings", "Diagnostics"])
    with settings_tab:
        input_mode = st.radio(
            "Input mode",
            ["Genius lyrics", "CSV upload mode", "Paste lyrics mode"],
            index=0,
        )
        st.text_input("Artist name", key="single_artist_name_input", placeholder="e.g. Taylor Swift")
        max_songs = int(
            st.number_input(
                "Max songs",
                min_value=10,
                max_value=500,
                value=50,
                step=5,
            )
        )
        include_features = st.checkbox("Include featured songs", value=False)
        dedupe_mode = st.radio(
            "Deduping mode",
            [
                "Strict canonical songs only",
                "Keep major alternate versions",
                "Keep everything",
            ],
            index=0,
        )
        st.caption("Version exclusions (live/remix/demo/acoustic/translation) are applied automatically in Strict canonical mode.")
        filtering_strictness_label = st.selectbox(
            "Filtering strictness",
            options=["Basic", "Lyric-clean", "Theme-focused"],
            index=1,
        )
        categorize_words = st.checkbox("Categorize top words with OpenAI", value=False)
        show_wordcloud = st.checkbox("Show word cloud", value=True)
        uploaded_csv = None
        if input_mode == "CSV upload mode":
            uploaded_csv = st.file_uploader(
                "CSV file (title, artist, lyrics required)",
                type=["csv"],
                help="Optional columns: album, release_date, url",
            )
        parallel_lyric_fetch_workers = st.slider(
            "Parallel lyric fetch workers",
            min_value=1,
            max_value=100,
            value=2,
            help="Higher values may be faster but more likely to trigger Genius rate limits.",
        )
    with diagnostics_tab:
        show_filtered_out_diagnostics = st.checkbox("Show filtered-out words diagnostics", value=False)
        show_lyricsgenius_diagnostics = st.checkbox("Show lyricsgenius diagnostics", value=False)
        show_archive_explorer = st.checkbox("Show archive explorer", value=False)
    with settings_tab:
        run_clicked = st.button("Run analysis", type="primary")

    artist_name = str(st.session_state.get("single_artist_name_input", "") or "").strip()
    genius_suggestion_pending = bool(st.session_state.pop("single_artist_run_pending", False))
    if input_mode != "Genius lyrics":
        genius_suggestion_pending = False

    strictness_map: dict[str, StrictnessMode] = {
        "Basic": "basic",
        "Lyric-clean": "lyric_clean",
        "Theme-focused": "theme_focused",
    }
    filtering_strictness = strictness_map[filtering_strictness_label]
    exclude_versions = dedupe_mode == "Strict canonical songs only"

    if input_mode == "Genius lyrics":
        _render_genius_lyrics_dismissible_callouts()

    if input_mode == "Paste lyrics mode":
        _render_paste_mode_inputs()

    if "corpus_songs" not in st.session_state:
        st.session_state["corpus_songs"] = []
        st.session_state["corpus_source_used"] = ""
        st.session_state["official_api_exact_results"] = []
    if "workflow_stage" not in st.session_state:
        st.session_state["workflow_stage"] = "idle"

    if not (run_clicked or genius_suggestion_pending) and _sidebar_drifted_from_loaded_corpus(
        input_mode=input_mode,
        artist_name=artist_name,
        max_songs=max_songs,
        include_features=include_features,
    ):
        _clear_single_artist_corpus_session()
        st.info("Artist, max songs, or input mode changed — click **Run analysis** to load the new search.")

    if run_clicked or genius_suggestion_pending:
        st.session_state["archived_analysis_row"] = None
        st.session_state["archive_decision"] = None
        st.session_state["workflow_stage"] = "idle"
        st.session_state.pop(_SINGLE_ARTIST_SUPABASE_SAVE_PENDING, None)
        with st.status("Preparing corpus...", expanded=True):
            try:
                if input_mode != "Genius lyrics":
                    raw_songs, source_used = _collect_raw_songs(
                        input_mode=input_mode,
                        artist_name=artist_name,
                        max_songs=max_songs,
                        include_features=include_features,
                        uploaded_csv=uploaded_csv,
                        auto_hydrate_workers=parallel_lyric_fetch_workers,
                    )
                    st.session_state["corpus_songs"] = raw_songs
                    st.session_state["corpus_source_used"] = source_used
                    st.session_state["workflow_stage"] = "ready_to_analyze"
                    st.session_state[_SINGLE_ARTIST_SUPABASE_SAVE_PENDING] = True
                else:
                    payload = fetch_artist_songs(
                        artist_name=artist_name,
                        max_songs=max_songs,
                        include_features=include_features,
                    )
                    if not payload.get("songs"):
                        typed = artist_name.strip()
                        suggestions = suggest_similar_artists(typed) if typed else []
                        raise ArtistNotFoundError(typed, suggestions)
                    st.session_state["artist_metadata"] = {
                        "artist_name": payload.get("artist_name") or artist_name.strip(),
                        "artist_id": payload.get("artist_id"),
                        "slug": slugify_artist_name(str(payload.get("artist_name") or artist_name)),
                    }
                    st.session_state["candidate_songs"] = list(payload.get("songs", []))[:max_songs]
                    st.session_state["workflow_stage"] = "metadata_loaded"
            except MissingGeniusTokenError as exc:
                st.error(str(exc))
                return
            except ArtistNotFoundError as exc:
                st.error(
                    f"No songs found on Genius for **{exc.typed_name!r}**. "
                    "Check spelling or try a name that matches Genius exactly."
                )
                if exc.suggestions and input_mode == "Genius lyrics":
                    _render_clickable_artist_suggestions(
                        exc.suggestions,
                        session_key_pending="single_artist_run_pending",
                        input_session_key="single_artist_name_input",
                        widget_key_prefix="pick_suggested_artist",
                    )
                elif exc.suggestions:
                    st.caption("Did you mean one of these?")
                    st.write(", ".join(exc.suggestions))
                return
            except GeniusCloudflareChallengeError:
                st.error(
                    "Genius blocked this web request with a Cloudflare challenge. "
                    "This usually affects unofficial Genius web endpoints, not necessarily your API key. "
                    "Try official metadata mode or CSV upload mode."
                )
                return
            except GeniusRequestError as exc:
                st.error(f"Genius request failed after retries: {exc}")
                return
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to load songs: {exc}")
                return
        st.session_state["corpus_artist_name"] = artist_name
        st.session_state["corpus_max_songs"] = int(max_songs)
        st.session_state["corpus_include_features"] = include_features
        st.session_state["corpus_input_mode"] = input_mode

    if input_mode == "Genius lyrics" and st.session_state.get("workflow_stage") in {
        "metadata_loaded",
        "archive_found",
        "checking_r2",
        "r2_loaded",
        "hydrating_lyrics",
    }:
        artist_meta = dict(st.session_state.get("artist_metadata") or {})
        candidate_songs = list(st.session_state.get("candidate_songs") or [])
        if not candidate_songs:
            st.error("No candidate songs found for this artist.")
            return
        settings_for_archive = {
            "filtering_strictness": filtering_strictness,
            "dedupe_mode": dedupe_mode,
            "include_features": include_features,
        }
        genius_artist_id = str(artist_meta.get("artist_id") or "")
        if st.session_state["workflow_stage"] == "metadata_loaded":
            archived = get_archived_analysis_for_artist(genius_artist_id, settings_for_archive) if genius_artist_id else None
            st.session_state["archived_analysis_row"] = archived
            st.session_state["workflow_stage"] = "archive_found" if archived else "checking_r2"

        if st.session_state["workflow_stage"] == "archive_found":
            archived = st.session_state.get("archived_analysis_row")
            if archived:
                st.subheader("Supabase Archive Check")
                st.info("Archived analysis found")
                archive_cols = st.columns(4)
                archive_cols[0].metric(
                    "Songs analyzed",
                    int(archived.get("songs_analyzed") or 0),
                    help=SONGS_ANALYZED_METRIC_HELP,
                )
                archive_cols[1].metric("Meaningful tokens", int(archived.get("total_meaningful_tokens") or 0))
                archive_cols[2].metric("Unique words", int(archived.get("unique_word_count") or 0))
                archive_cols[3].metric("Categories", "yes" if bool(archived.get("categories_exist")) else "no")
                st.caption(
                    f"Artist: {archived.get('artist_name')} | Last analyzed: {archived.get('created_at')} | "
                    f"Filtering: {archived.get('filtering_strictness')} | Dedupe: {archived.get('dedupe_mode')}"
                )
                b1, b2 = st.columns(2)
                if b1.button("Load archived analysis", key="load_archived_analysis_btn"):
                    st.session_state["archive_decision"] = "load"
                    st.session_state["workflow_stage"] = "archive_loaded"
                if b2.button("Run fresh (R2 + lyricsgenius)", key="run_fresh_instead_btn"):
                    st.session_state["archive_decision"] = "fresh"
                    st.session_state["workflow_stage"] = "checking_r2"
            else:
                st.session_state["workflow_stage"] = "checking_r2"

        if st.session_state.get("workflow_stage") == "archive_loaded":
            archived = st.session_state.get("archived_analysis_row")
            if archived:
                _render_archived_analysis_block(
                    dict(archived),
                    show_wordcloud=show_wordcloud,
                    show_filtered_out_diagnostics=show_filtered_out_diagnostics,
                    show_lyricsgenius_diagnostics=show_lyricsgenius_diagnostics,
                    widget_namespace="single_genius_tab",
                )
                return

        if st.session_state["workflow_stage"] == "checking_r2":
            r2_key = get_r2_corpus_key(artist_meta)
            st.session_state["r2_corpus_key"] = r2_key
            exists = r2_object_exists(r2_key)
            st.subheader("R2 Corpus Check")
            if exists:
                corpus = download_json_from_r2(r2_key) or make_empty_corpus(artist_meta)
                summary = get_corpus_summary(corpus)
                st.caption(
                    f"Found lyric corpus in R2 | songs_with_lyrics={summary['songs_with_lyrics']} | "
                    f"song_count={summary['song_count']} | updated_at={summary['updated_at']}"
                )
            else:
                corpus = make_empty_corpus(artist_meta)
                st.caption("No lyric corpus in R2 yet. Starting an empty corpus for this artist.")
            st.session_state["r2_corpus"] = corpus
            st.session_state["workflow_stage"] = "r2_loaded"

        if st.session_state["workflow_stage"] == "r2_loaded":
            corpus = dict(st.session_state.get("r2_corpus") or {})
            r2_songs = list(corpus.get("songs", [])) if isinstance(corpus.get("songs", []), list) else []
            merged = _merge_candidate_with_corpus(candidate_songs, r2_songs)
            merged_cap = merged[:max_songs]
            with_lyrics_count = sum(1 for s in merged_cap if int(s.get("lyrics_char_count", 0) or 0) > 0)
            if with_lyrics_count >= max_songs:
                st.session_state["corpus_songs"] = merged_cap
                st.session_state["corpus_source_used"] = "r2_cache"
                st.session_state["workflow_stage"] = "ready_to_analyze"
                st.session_state[_SINGLE_ARTIST_SUPABASE_SAVE_PENDING] = True
            else:
                st.session_state["corpus_songs"] = merged_cap
                st.session_state["corpus_source_used"] = "r2_plus_lyricsgenius"
                st.session_state["workflow_stage"] = "hydrating_lyrics"

        if st.session_state["workflow_stage"] == "hydrating_lyrics":
            st.subheader("Lyrics Hydration")
            st.caption(
                "If Genius blocks requests partway through, anything fetched so far is kept. "
                "You can open **Single artist** analysis on partial lyrics and run hydration again later."
            )
            corpus = list(st.session_state.get("corpus_songs") or [])
            missing_target_keys = {
                _song_key(song) for song in corpus if int(song.get("lyrics_char_count", 0) or 0) == 0
            }
            try:
                hydrated, diagnostics_rows = hydrate_missing_lyrics_with_lyricsgenius(
                    corpus_songs=corpus,
                    max_workers=parallel_lyric_fetch_workers,
                    target_song_keys=missing_target_keys,
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(
                    f"Hydration raised an error ({exc!r}). "
                    "Continuing with any lyrics updated so far — you can still run analysis or retry hydration."
                )
                hydrated = corpus
                diagnostics_rows = list(st.session_state.get("lyricsgenius_hydration_diagnostics") or [])
            st.session_state["lyricsgenius_hydration_diagnostics"] = diagnostics_rows
            hydrated_cap = hydrated[:max_songs]
            st.session_state["corpus_songs"] = hydrated_cap
            merged_corpus = dict(st.session_state.get("r2_corpus") or make_empty_corpus(artist_meta))
            merged_corpus["songs"] = hydrated_cap
            try:
                upload_json_to_r2(str(st.session_state.get("r2_corpus_key")), merged_corpus)
                summ = get_corpus_summary(merged_corpus)
                _flash_notice(
                    f"Lyric corpus saved to R2 ({summ['songs_with_lyrics']}/{summ['song_count']} songs with lyrics)."
                )
            except Exception as r2_exc:  # noqa: BLE001
                st.warning(
                    f"Could not upload corpus to R2 ({r2_exc!r}). "
                    "Hydrated lyrics in this session are still available — run analysis or fix R2 and retry."
                )
            st.session_state["workflow_stage"] = "ready_to_analyze"
            st.session_state[_SINGLE_ARTIST_SUPABASE_SAVE_PENDING] = True
            st.session_state["corpus_source_used"] = (
                "lyricsgenius_fresh"
                if len(list((st.session_state.get("r2_corpus") or {}).get("songs", []))) == 0
                else "r2_plus_lyricsgenius"
            )

    if input_mode == "Genius lyrics" and st.session_state.get("workflow_stage") in {
        "metadata_loaded",
        "archive_found",
        "checking_r2",
        "r2_loaded",
        "hydrating_lyrics",
    }:
        return

    corpus_songs: list[dict[str, object]] = st.session_state.get("corpus_songs", [])
    source_used = st.session_state.get("corpus_source_used", input_mode)

    if not corpus_songs:
        st.info("Load songs first: choose an input mode and click Run analysis.")
        return

    if source_used == "Genius lyrics":
        st.caption(
            "Hydration only requests songs still missing lyrics. If Genius blocks mid-run, progress so far is kept — "
            "run analysis on partial data or hydrate again after a pause."
        )
        if st.button("Hydrate metadata songs with lyricsgenius"):
            with st.spinner("Fetching lyrics from Genius pages…"):
                try:
                    hydrated, diagnostics_rows = hydrate_missing_lyrics_with_lyricsgenius(
                        corpus_songs=corpus_songs,
                        max_workers=parallel_lyric_fetch_workers,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.warning(
                        f"Hydration raised an error ({exc!r}). "
                        "Any lyrics updated before the failure are kept — you can still run analysis or retry."
                    )
                    hydrated = list(corpus_songs)
                    diagnostics_rows = list(st.session_state.get("lyricsgenius_hydration_diagnostics") or [])
            cap = max(1, int(max_songs))
            hydrated_cap = hydrated[:cap]
            st.session_state["corpus_songs"] = hydrated_cap
            st.session_state["lyricsgenius_hydration_diagnostics"] = diagnostics_rows
            st.session_state["corpus_max_songs"] = cap
            corpus_songs = hydrated_cap
            st.session_state[_SINGLE_ARTIST_SUPABASE_SAVE_PENDING] = True

    _sync_lyrics_char_counts_from_lyrics_body(corpus_songs)

    corpus_df = pd.DataFrame(corpus_songs)
    hydration_rows = st.session_state.get("lyricsgenius_hydration_diagnostics", [])

    if not corpus_df.empty:
        if "lyrics_char_count" not in corpus_df.columns:
            corpus_df["lyrics_char_count"] = corpus_df.get("lyrics", "").apply(_lyrics_char_count)
        if "lyrics_status" not in corpus_df.columns:
            corpus_df["lyrics_status"] = corpus_df["lyrics_char_count"].apply(
                lambda n: "available" if int(n) > 0 else "missing"
            )
        if "lyrics_source" not in corpus_df.columns:
            corpus_df["lyrics_source"] = corpus_df["lyrics_char_count"].apply(
                lambda n: "available" if int(n) > 0 else "none"
            )

    preview_dedupe = dedupe_songs(
        songs=corpus_songs,
        mode=dedupe_mode,
        exclude_versions=exclude_versions,
        canonical_artist_name=artist_name if artist_name.strip() else None,
    )
    included_preview = preview_dedupe.included
    excluded_preview = preview_dedupe.excluded
    manually_included_song_keys: list[str] = []
    manually_excluded_song_keys: list[str] = []
    if included_preview:
        with st.expander("Included songs controls", expanded=False):
            included_preview_df = pd.DataFrame(included_preview)
            if not included_preview_df.empty:
                st.caption("Select rows in the table to exclude from analysis.")
                included_preview_df["exclude_from_analysis"] = False
                included_preview_df["_selection_key"] = included_preview_df.apply(
                    lambda row: _song_selection_key(row.to_dict()),
                    axis=1,
                )
                edited_included = st.data_editor(
                    included_preview_df[
                        [
                            "exclude_from_analysis",
                            "title",
                            "artist",
                            "album",
                            "release_date",
                            "_selection_key",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                    disabled=["title", "artist", "album", "release_date", "_selection_key"],
                    column_config={"_selection_key": None},
                    key="included_controls_table",
                )
                manually_excluded_song_keys = [
                    str(row["_selection_key"])
                    for _, row in edited_included.iterrows()
                    if bool(row.get("exclude_from_analysis", False))
                ]
    if excluded_preview:
        with st.expander("Excluded songs controls", expanded=False):
            excluded_preview_df = pd.DataFrame(excluded_preview)
            if not excluded_preview_df.empty:
                st.caption("Select rows in the table to include in analysis.")
                excluded_preview_df["include_in_analysis"] = False
                excluded_preview_df["_selection_key"] = excluded_preview_df.apply(
                    lambda row: _song_selection_key(row.to_dict()),
                    axis=1,
                )
                edited_excluded = st.data_editor(
                    excluded_preview_df[
                        [
                            "include_in_analysis",
                            "title",
                            "artist",
                            "exclude_reason",
                            "album",
                            "release_date",
                            "_selection_key",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                    disabled=["title", "artist", "exclude_reason", "album", "release_date", "_selection_key"],
                    column_config={"_selection_key": None},
                    key="excluded_controls_table",
                )
                manually_included_song_keys = [
                    str(row["_selection_key"])
                    for _, row in edited_excluded.iterrows()
                    if bool(row.get("include_in_analysis", False))
                ]

    analysis_result, included_songs, excluded_songs, token_diagnostics = _run_analysis_pipeline(
        raw_songs=corpus_songs,
        dedupe_mode=dedupe_mode,
        exclude_versions=exclude_versions,
        filtering_strictness=filtering_strictness,
        manually_included_song_keys=set(manually_included_song_keys),
        manually_excluded_song_keys=set(manually_excluded_song_keys),
        canonical_artist_name=artist_name if artist_name.strip() else None,
    )
    if analysis_result is None:
        st.warning("No songs remained after filtering/deduplication.")
        return

    total_lyrics_chars = sum(int(s.get("lyrics_char_count", 0) or 0) for s in included_songs)
    total_raw_words_before_clean = sum(len(str(s.get("lyrics", "") or "").split()) for s in included_songs)

    display_title = st.session_state.get("corpus_display_title")
    if not display_title:
        if source_used == "Genius lyrics":
            display_title = artist_name.strip() or "Genius artist"
        else:
            display_title = "Analysis"
    st.markdown(f"## {display_title}")
    if source_used == "Genius lyrics" and st.session_state.get("hydration_fetch_status"):
        st.caption(str(st.session_state["hydration_fetch_status"]))

    st.subheader("Analysis Metrics")
    debug_cols = st.columns(5)
    debug_cols[0].metric(
        "Total songs in corpus",
        len(corpus_songs),
        help="Songs in the loaded corpus before dedupe / manual exclude rules.",
    )
    debug_cols[1].metric(
        "Total lyrics chars (analyzed)",
        total_lyrics_chars,
        help="Sum of lyric lengths for songs included in analysis.",
    )
    debug_cols[2].metric(
        "Raw words before cleaning (analyzed)",
        total_raw_words_before_clean,
        help="Whitespace-split counts on included songs only.",
    )
    debug_cols[3].metric("Meaningful words after cleaning", int(analysis_result.summary["total_meaningful_tokens"]))
    debug_cols[4].metric("Requested song count", int(max_songs))

    summary = analysis_result.summary
    st.caption(f"Source used: {source_used}")
    st.caption(
        f"Filtering mode: {summary.get('filtering_strictness', filtering_strictness)} "
        f"(stopword version: {summary.get('stopword_version', STOPWORD_VERSION)})"
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Songs analyzed",
        int(summary["total_songs_analyzed"]),
        help=SONGS_ANALYZED_METRIC_HELP,
    )
    metric_cols[1].metric("Unique words", int(summary["unique_vocabulary_size"]))
    metric_cols[2].metric("Lexical diversity", f"{summary['lexical_diversity']:.3f}")
    metric_cols[3].metric("Meaningful tokens", int(summary["total_meaningful_tokens"]))

    st.subheader("Top words")
    st.plotly_chart(
        make_top_words_chart(analysis_result.word_frequencies_df),
        width="stretch",
        key="plt_single_top_words",
    )
    with st.expander("Top 100 words (copyable text)", expanded=False):
        top_words_lines = [
            f"{idx + 1}. {row['word']} ({int(row['count'])})"
            for idx, (_, row) in enumerate(analysis_result.word_frequencies_df.head(100).iterrows())
        ]
        st.text("\n".join(top_words_lines))

    categories_payload = None
    if categorize_words:
        with st.spinner("Categorizing top words with OpenAI..."):
            categories_payload, warning = categorize_top_words(analysis_result.top_words[:100])
        if warning:
            st.warning(warning)
        elif categories_payload:
            categories_payload = _sort_categories_payload(categories_payload)
            st.subheader("Semantic categories")
            cat_chart = make_category_chart(categories_payload)
            if cat_chart is not None:
                st.plotly_chart(cat_chart, width="stretch", key="plt_single_categories")
            with st.expander("Raw categories JSON", expanded=False):
                st.json(categories_payload)

    if input_mode == "Genius lyrics":
        if st.session_state.get(_SINGLE_ARTIST_SUPABASE_SAVE_PENDING):
            try:
                artist_meta = dict(st.session_state.get("artist_metadata") or {})
                settings_payload = {
                    "filtering_strictness": filtering_strictness,
                    "dedupe_mode": dedupe_mode,
                    "include_features": include_features,
                    "requested_song_count": int(max_songs),
                    "source_used": st.session_state.get("corpus_source_used", source_used),
                }
                saved = save_analysis_to_supabase(
                    artist_metadata=artist_meta,
                    analysis_results=_build_analysis_results_payload(
                        summary=summary,
                        word_df=analysis_result.word_frequencies_df,
                        bigram_df=analysis_result.bigram_frequencies_df,
                        categories_payload=categories_payload,
                    ),
                    settings=settings_payload,
                    r2_corpus_key=str(st.session_state.get("r2_corpus_key") or ""),
                )
                st.session_state.pop(_SINGLE_ARTIST_SUPABASE_SAVE_PENDING, None)
                st.session_state["workflow_stage"] = "analysis_complete"
                if saved.get("__lyric_atlas_save_action__") == "retained_existing":
                    _flash_notice(
                        "Supabase left unchanged — an existing archive already analyzed more songs for this filter/dedupe/featured/source.",
                        icon="ℹ️",
                    )
                else:
                    _flash_notice(f"Saved to Supabase (run {saved.get('analysis_run_id', '')}).")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not save to Supabase: {exc}")
        else:
            st.session_state["workflow_stage"] = "analysis_complete"

    st.subheader("Top bigrams")
    st.plotly_chart(
        make_top_bigrams_chart(analysis_result.bigram_frequencies_df),
        width="stretch",
        key="plt_single_bigrams",
    )

    if show_filtered_out_diagnostics:
        st.subheader("Filtered-out words diagnostics")
        diag_cols = st.columns(5)
        diag_cols[0].metric("Total raw tokens", int(summary["total_tokens_before_filtering"]))
        diag_cols[1].metric(
            "Removed by standard stopwords",
            int(summary.get("removed_standard_stopwords", 0)),
        )
        diag_cols[2].metric(
            "Removed by definite lyric stopwords",
            int(summary.get("removed_definite_lyric_stopwords", 0)),
        )
        diag_cols[3].metric(
            "Removed by maybe lyric stopwords",
            int(summary.get("removed_maybe_lyric_stopwords", 0)),
        )
        diag_cols[4].metric("Meaningful tokens remaining", int(summary["total_meaningful_tokens"]))
        removed_top = token_diagnostics.get("removed_words_top_50", [])
        if removed_top:
            st.dataframe(pd.DataFrame(removed_top), width="stretch")
        else:
            st.info("No removed tokens to display for current settings.")
    if show_lyricsgenius_diagnostics:
        st.subheader("lyricsgenius diagnostics")
        if hydration_rows:
            st.dataframe(pd.DataFrame(hydration_rows), width="stretch")
        else:
            st.info("No lyricsgenius diagnostics available yet. Run hydration first.")
    if show_archive_explorer:
        _render_archive_explorer(dict(st.session_state.get("artist_metadata") or {}))

    if show_wordcloud:
        st.subheader("Word cloud")
        wc_top_n = st.slider(
            "Top words to include in the cloud",
            min_value=5,
            max_value=200,
            value=25,
            step=5,
            key="single_wordcloud_top_n",
            help="Uses the same ranked frequencies as **Top words**; default 25 matches the bar chart.",
        )
        image = make_wordcloud_image(analysis_result.word_frequencies_df, top_n=int(wc_top_n))
        if image is not None:
            st.image(image, width="stretch")
        else:
            st.info("No word frequencies available for the cloud.")

    with st.expander("Included songs", expanded=False):
        included_df = pd.DataFrame(included_songs)
        if not included_df.empty:
            songs_to_show = st.radio(
                "Included songs rows to display",
                options=[10, 50, 100],
                horizontal=True,
                index=0,
            )
            if "lyrics_char_count" not in included_df.columns:
                included_df["lyrics_char_count"] = included_df.get("lyrics", "").apply(_lyrics_char_count)
            if "lyrics_status" not in included_df.columns:
                included_df["lyrics_status"] = included_df["lyrics_char_count"].apply(
                    lambda n: "available" if int(n) > 0 else "missing"
                )
            if "lyrics_source" not in included_df.columns:
                included_df["lyrics_source"] = included_df["lyrics_char_count"].apply(
                    lambda n: "available" if int(n) > 0 else "none"
                )
            if hydration_rows:
                hydration_df = pd.DataFrame(hydration_rows)
                hydration_df["join_title"] = hydration_df["input_title"].astype(str).str.strip().str.lower()
                hydration_df["join_artist"] = hydration_df["input_artist"].astype(str).str.strip().str.lower()
                latest_hydration = (
                    hydration_df.sort_index()
                    .drop_duplicates(subset=["join_title", "join_artist"], keep="last")
                    [
                        [
                            "join_title",
                            "join_artist",
                            "found_song",
                            "returned_title",
                            "returned_artist",
                            "lyricsgenius_lyrics_char_count",
                            "blocked",
                            "error_message",
                        ]
                    ]
                )
                included_df["join_title"] = included_df["title"].astype(str).str.strip().str.lower()
                included_df["join_artist"] = included_df["artist"].astype(str).str.strip().str.lower()
                included_df = included_df.merge(
                    latest_hydration,
                    on=["join_title", "join_artist"],
                    how="left",
                )
            requested_columns = [
                "title",
                "artist",
                "lyrics_status",
                "lyrics_char_count",
                "lyrics_source",
                "found_song",
                "returned_title",
                "returned_artist",
                "lyricsgenius_lyrics_char_count",
                "blocked",
                "error_message",
                "album",
                "release_date",
                "url",
            ]
            available_columns = [col for col in requested_columns if col in included_df.columns]
            st.dataframe(included_df[available_columns].head(int(songs_to_show)))
        else:
            st.info("No included songs to display.")

    if excluded_songs:
        with st.expander("Excluded songs", expanded=False):
            st.dataframe(excluded_songs, width="stretch")

    st.subheader("Exports")
    export_cols = st.columns(3)
    words_export_df = analysis_result.word_frequencies_df.copy()
    words_export_df["filtering_strictness"] = str(summary.get("filtering_strictness", filtering_strictness))
    words_export_df["stopword_version"] = str(summary.get("stopword_version", STOPWORD_VERSION))
    bigrams_export_df = analysis_result.bigram_frequencies_df.copy()
    bigrams_export_df["filtering_strictness"] = str(summary.get("filtering_strictness", filtering_strictness))
    bigrams_export_df["stopword_version"] = str(summary.get("stopword_version", STOPWORD_VERSION))
    analysis_settings = {
        "input_mode": input_mode,
        "dedupe_mode": dedupe_mode,
        "exclude_versions": exclude_versions,
        "filtering_strictness": str(summary.get("filtering_strictness", filtering_strictness)),
        "stopword_version": str(summary.get("stopword_version", STOPWORD_VERSION)),
    }
    export_cols[0].download_button(
        "Download words CSV",
        data=words_export_df.to_csv(index=False).encode("utf-8"),
        file_name="word_frequencies.csv",
        mime="text/csv",
    )
    export_cols[1].download_button(
        "Download bigrams CSV",
        data=bigrams_export_df.to_csv(index=False).encode("utf-8"),
        file_name="bigram_frequencies.csv",
        mime="text/csv",
    )
    export_cols[2].download_button(
        "Download analysis JSON",
        data=_safe_json(
            {
                "summary": summary,
                "top_words": analysis_result.top_words,
                "categories": categories_payload,
                "analysis_settings": analysis_settings,
            }
        ),
        file_name="lyric_atlas_analysis.json",
        mime="application/json",
    )


def _archives_browse_preview_rows(archives: list[dict[str, object]]) -> pd.DataFrame:
    rows_out: list[dict[str, object]] = []
    for r in archives:
        sj = dict(r.get("settings_json") or {})
        rows_out.append(
            {
                "Artist": r.get("artist_name"),
                "Songs analyzed": int(r.get("songs_analyzed") or 0),
                "Last updated": str(r.get("created_at") or "")[:19],
                "Deduping": r.get("dedupe_mode"),
                "Filtering": r.get("filtering_strictness"),
                "Featured songs": "Yes" if sj.get("include_features") else "No",
                "Requested #": sj.get("requested_song_count"),
                "Run ID": r.get("analysis_run_id"),
            }
        )
    return pd.DataFrame(rows_out)


def _render_archived_artists_tab() -> None:
    """Search and load Supabase analysis archives (per-artist de-dupe by filter/dedupe/features/source)."""
    st.subheader("View archived artists")
    st.caption(
        "Search saved runs from Supabase. For each artist we keep one row per "
        "**filtering + deduping + featured + source** combo — the one with the **most songs analyzed**; "
        "weaker runs are deleted after saves, and new saves are skipped when a larger run already exists. "
        "Requested song count is still stored for reference. "
        "**Select one row** in the table, then click **Load selected archive**."
    )
    q = st.text_input(
        "Search by artist name",
        key="archive_browse_query",
        placeholder="e.g. Gunna (leave empty for most recent across all artists)",
    )
    max_list = int(
        st.number_input(
            "Maximum archives to list",
            min_value=5,
            max_value=200,
            value=25,
            step=5,
            help="Fetched from Supabase in most-recent-first order.",
            key="archive_browse_max_list",
        )
    )

    table_sig = f"{q.strip()}|{max_list}"
    if st.session_state.get("_archive_table_sig") != table_sig:
        st.session_state["_archive_table_sig"] = table_sig
        st.session_state["_archive_table_nonce"] = int(st.session_state.get("_archive_table_nonce", 0)) + 1
    table_key = f"archive_browse_sel_{st.session_state.get('_archive_table_nonce', 0)}"

    try:
        if q.strip():
            rows_raw = search_archives_by_artist_name(q.strip(), limit=max_list)
        else:
            rows_raw = list_recent_archives(limit=max_list)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load archives (check Supabase env and table `lyric_analysis_archives`): {exc}")
        return

    if not rows_raw:
        st.info("No matching archives yet. Run an analysis in **Single artist** to create one.")
        return

    preview_df = _archives_browse_preview_rows(rows_raw)
    st.caption("Click a row to select it (single-row selection).")
    st.dataframe(
        preview_df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
    )

    idx = _dataframe_single_selected_row_index(table_key)
    if idx is None:
        st.info("Select a row in the table above, then load.")
    elif idx < 0 or idx >= len(rows_raw):
        st.warning("Selection is out of date; try selecting the row again.")
        idx = None

    if st.button(
        "Load selected archive",
        type="primary",
        key="archive_browse_load_btn",
        disabled=idx is None,
    ):
        run_id = str(rows_raw[idx].get("analysis_run_id") or "") if idx is not None else ""
        row = load_analysis_from_supabase(run_id) if run_id else None
        if not row:
            st.warning("That archive could not be loaded.")
        else:
            st.session_state["archived_browse_loaded"] = row
            _flash_notice(f"Loaded archive {run_id}.", icon="📂")

    loaded = st.session_state.get("archived_browse_loaded")
    if loaded:
        st.divider()
        _render_archived_analysis_block(dict(loaded), widget_namespace="archived_artists_tab")


def main() -> None:
    """Lyric Atlas Streamlit entrypoint."""
    st.set_page_config(page_title="Lyric Atlas", layout="wide")
    st.title("Lyric Atlas")
    st.caption("Analyze the words an artist returns to most — with Supabase + R2 archives.")
    if "dismiss_genius_mode_info" not in st.session_state:
        st.session_state["dismiss_genius_mode_info"] = False
    if "dismiss_lyricsgenius_warning" not in st.session_state:
        st.session_state["dismiss_lyricsgenius_warning"] = False

    tab_single, tab_compare, tab_archives = st.tabs(
        ["Single artist", "Compare artists", "View archived artists"]
    )
    with tab_single:
        _render_single_artist_tab()
    with tab_compare:
        _render_compare_artists_tab()
    with tab_archives:
        _render_archived_artists_tab()


if __name__ == "__main__":
    main()
