"""Streamlit entrypoint for Artist Wordprint."""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv(".env.local", override=True)

from src.analysis import analyze_tokens
from src.categorize import categorize_top_words
from src.dedupe import dedupe_songs
from src.genius_client import (
    GeniusCloudflareChallengeError,
    GeniusRequestError,
    MissingGeniusTokenError,
    fetch_artist_songs,
)
from src.cache import save_artist_cache
from src.lyricsgenius_client import fetch_lyrics_with_diagnostics
from src.text_filtering import (
    STOPWORD_VERSION,
    StrictnessMode,
    TokenizationResult,
    aggregate_removed_tokens,
    tokenize_and_filter_lyrics,
)
from src.visualizations import (
    make_category_chart,
    make_top_bigrams_chart,
    make_top_words_chart,
    make_wordcloud_image,
)

CACHE_DIR = Path("data/cache")


def _safe_json(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def _lyrics_char_count(value: object) -> int:
    text = str(value or "")
    return len(text) if text else 0


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
    print(f"cached/existing lyrics char count: {len(existing_lyrics)}")
    print(f"lyricsgenius lyrics char count: {len(lg_lyrics)}")

    final_lyrics = lg_lyrics or existing_lyrics or official_lyrics or ""

    merged: dict[str, object] = {}
    merged.update(existing)
    merged.update(official)
    merged["lyrics"] = final_lyrics

    if lg_lyrics:
        merged["lyrics_source"] = "lyricsgenius"
    elif existing_lyrics:
        merged["lyrics_source"] = str(existing.get("lyrics_source", "cache"))
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
        cache_dir=CACHE_DIR,
    )
    st.session_state["official_api_exact_results"] = payload.get("songs", [])
    existing_lookup = {_song_key(song): song for song in st.session_state.get("corpus_songs", [])}
    songs: list[dict[str, object]] = []
    for official_song in payload.get("songs", []):
        official_base = dict(official_song)
        official_base["lyrics"] = str(official_base.get("lyrics", "") or "")
        official_base["lyrics_status"] = "available" if official_base["lyrics"] else "missing"
        official_base["lyrics_source"] = "cache" if official_base["lyrics"] else "none"
        official_base["lyrics_char_count"] = len(official_base["lyrics"])
        official_base["official_api_lyrics_char_count"] = len(str(official_song.get("lyrics", "") or ""))
        merged_song = merge_song_data(
            official_song=official_base,
            existing_song=existing_lookup.get(_song_key(official_base)),
            lyricsgenius_result=None,
        )
        songs.append(merged_song)
    return songs, "Official Genius metadata mode"


def _update_official_cache_with_songs(
    artist_name: str,
    songs: list[dict[str, object]],
    max_songs: int,
    include_features: bool,
) -> None:
    payload = {
        "artist_name": artist_name,
        "max_songs": max_songs,
        "include_features": include_features,
        "songs": songs,
    }
    save_artist_cache(CACHE_DIR, artist_name, payload)


def hydrate_missing_lyrics_with_lyricsgenius(
    corpus_songs: list[dict],
    max_workers: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Hydrate missing lyrics in-place using controlled parallel lyricsgenius calls."""
    diagnostics_rows: list[dict[str, object]] = []
    progress = st.progress(0.0)
    progress_text = st.empty()
    missing_candidates = [song for song in corpus_songs if int(song.get("lyrics_char_count", 0) or 0) == 0]
    if not missing_candidates:
        st.info("No missing lyrics to hydrate.")
        st.session_state["lyricsgenius_hydration_diagnostics"] = diagnostics_rows
        return corpus_songs, diagnostics_rows

    fetched_count = 0
    missing_count = 0
    blocked_count = 0
    error_count = 0
    processed = 0
    blocked_detected = False
    batch_size = max_workers * 2

    def _worker(song: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        time.sleep(random.uniform(0.5, 2.0))
        diagnostics = fetch_lyrics_with_diagnostics(
            title=str(song.get("title", "") or ""),
            artist=str(song.get("artist", "") or ""),
        )
        return song, diagnostics

    for start_idx in range(0, len(missing_candidates), batch_size):
        if blocked_detected:
            st.warning("Genius appears to be blocking requests. Reduce parallel workers or try again later.")
            break
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
                _update_official_cache_with_songs(
                    st.session_state.get("corpus_artist_name", ""),
                    corpus_songs,
                    int(st.session_state.get("corpus_max_songs", 50)),
                    bool(st.session_state.get("corpus_include_features", False)),
                )
            elif blocked:
                update_song_lyrics_in_corpus(corpus_songs, song, "", "none", blocked=True)
                blocked_count += 1
                blocked_detected = True
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

            progress.progress(processed / len(missing_candidates))
            progress_text.write(f"Fetched {processed} / {len(missing_candidates)} songs")

    st.session_state["lyricsgenius_hydration_diagnostics"] = diagnostics_rows

    if any(int(row.get("lyricsgenius_lyrics_char_count", 0) or 0) > 0 for row in diagnostics_rows):
        if not any(int(song.get("lyrics_char_count", 0) or 0) > 0 for song in corpus_songs):
            st.warning("BUG: lyricsgenius returned lyrics, but the hydrated lyrics were not written into corpus_songs.")
    st.info(
        f"Hydration summary: fetched_count={fetched_count}, missing_count={missing_count}, "
        f"blocked_count={blocked_count}, error_count={error_count}"
    )
    return corpus_songs, diagnostics_rows


def main() -> None:
    """Render and run the Artist Wordprint app."""
    st.set_page_config(page_title="Artist Wordprint", layout="wide")
    st.title("Artist Wordprint")
    st.caption("Analyze the words an artist returns to most.")
    if "dismiss_genius_mode_info" not in st.session_state:
        st.session_state["dismiss_genius_mode_info"] = False
    if "dismiss_lyricsgenius_warning" not in st.session_state:
        st.session_state["dismiss_lyricsgenius_warning"] = False

    with st.sidebar:
        st.header("Controls")
        settings_tab, diagnostics_tab = st.tabs(["Settings", "Diagnostics"])
    with settings_tab:
        input_mode = st.radio(
            "Input mode",
            ["Genius lyrics", "CSV upload mode", "Paste lyrics mode"],
            index=0,
        )
        artist_name = st.text_input("Artist name", value="")
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
        dedupe_mode = st.selectbox(
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
    with settings_tab:
        run_clicked = st.button("Run analysis", type="primary")

    strictness_map: dict[str, StrictnessMode] = {
        "Basic": "basic",
        "Lyric-clean": "lyric_clean",
        "Theme-focused": "theme_focused",
    }
    filtering_strictness = strictness_map[filtering_strictness_label]
    exclude_versions = dedupe_mode == "Strict canonical songs only"

    if input_mode == "Genius lyrics":
        if not st.session_state["dismiss_genius_mode_info"]:
            info_col, dismiss_info_col = st.columns([20, 1])
            with info_col:
                st.info(
                    "Genius lyrics mode uses `https://api.genius.com` metadata plus optional lyricsgenius hydration. "
                    "It may not include full lyrics; CSV/paste modes are the most reliable for lexical analysis."
                )
            with dismiss_info_col:
                if st.button("x", key="dismiss_genius_mode_info_btn", help="Dismiss this message"):
                    st.session_state["dismiss_genius_mode_info"] = True
                    st.rerun()
        if not st.session_state["dismiss_lyricsgenius_warning"]:
            warning_col, dismiss_warning_col = st.columns([20, 1])
            with warning_col:
                st.warning(
                    "lyricsgenius retrieves lyrics by scraping Genius song pages, so it may fail if Genius blocks automated "
                    "requests. The official Genius API is still used for metadata."
                )
            with dismiss_warning_col:
                if st.button("x", key="dismiss_lyricsgenius_warning_btn", help="Dismiss this message"):
                    st.session_state["dismiss_lyricsgenius_warning"] = True
                    st.rerun()

    if input_mode == "Paste lyrics mode":
        _render_paste_mode_inputs()

    if "corpus_songs" not in st.session_state:
        st.session_state["corpus_songs"] = []
        st.session_state["corpus_source_used"] = ""
        st.session_state["official_api_exact_results"] = []

    if run_clicked:
        with st.status("Preparing corpus...", expanded=True):
            try:
                raw_songs, source_used = _collect_raw_songs(
                    input_mode=input_mode,
                    artist_name=artist_name,
                    max_songs=max_songs,
                    include_features=include_features,
                    uploaded_csv=uploaded_csv,
                )
                if source_used == "Official Genius metadata mode":
                    print("Official API metadata loaded")
                else:
                    st.session_state["official_api_exact_results"] = []
            except MissingGeniusTokenError as exc:
                st.error(str(exc))
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

            st.session_state["corpus_songs"] = raw_songs
            st.session_state["corpus_source_used"] = "Genius lyrics" if source_used == "Official Genius metadata mode" else source_used
            st.session_state["corpus_artist_name"] = artist_name
            st.session_state["corpus_max_songs"] = max_songs
            st.session_state["corpus_include_features"] = include_features

    corpus_songs: list[dict[str, object]] = st.session_state.get("corpus_songs", [])
    source_used = st.session_state.get("corpus_source_used", input_mode)

    if not corpus_songs:
        st.info("Load songs first: choose an input mode and click Run analysis.")
        return

    if source_used == "Genius lyrics" and st.button("Hydrate metadata songs with lyricsgenius"):
        hydrated, diagnostics_rows = hydrate_missing_lyrics_with_lyricsgenius(
            corpus_songs=corpus_songs,
            max_workers=parallel_lyric_fetch_workers,
        )
        st.session_state["corpus_songs"] = hydrated
        st.session_state["lyricsgenius_hydration_diagnostics"] = diagnostics_rows
        corpus_songs = hydrated
        _update_official_cache_with_songs(artist_name, corpus_songs, max_songs, include_features)

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
            corpus_df["lyrics_source"] = corpus_df["lyrics_char_count"].apply(lambda n: "cache" if int(n) > 0 else "none")

    songs_with_lyrics = [s for s in corpus_songs if int(s.get("lyrics_char_count", 0) or 0) > 0]
    total_lyrics_chars = sum(int(s.get("lyrics_char_count", 0) or 0) for s in corpus_songs)
    total_raw_words_before_clean = sum(len(str(s.get("lyrics", "") or "").split()) for s in corpus_songs)

    if not songs_with_lyrics:
        st.warning(
            "No lyrics available. Official Genius API metadata does not include lyrics. "
            "Click 'Hydrate metadata songs with lyricsgenius' first."
        )
        return

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
            st.caption("Songs currently included after dedupe/version filtering. Select any to exclude from analysis.")
            options = [_song_selection_key(song) for song in included_preview]
            labels = {_song_selection_key(song): _song_display_label(song) for song in included_preview}
            manually_excluded_song_keys = st.multiselect(
                "Exclude from included songs",
                options=options,
                format_func=lambda key: labels.get(key, key),
                key="manually_excluded_song_keys",
            )
            included_preview_df = pd.DataFrame(included_preview)
            if not included_preview_df.empty:
                show_cols = [col for col in ["title", "artist", "album", "release_date"] if col in included_preview_df.columns]
                st.dataframe(included_preview_df[show_cols], width="stretch")
    if excluded_preview:
        with st.expander("Excluded songs controls", expanded=False):
            st.caption("Songs excluded by dedupe/version filters. Select any to include in analysis.")
            options = [_song_selection_key(song) for song in excluded_preview]
            labels = {_song_selection_key(song): _song_display_label(song) for song in excluded_preview}
            manually_included_song_keys = st.multiselect(
                "Include from excluded songs",
                options=options,
                format_func=lambda key: labels.get(key, key),
                key="manually_included_song_keys",
            )
            excluded_preview_df = pd.DataFrame(excluded_preview)
            if not excluded_preview_df.empty:
                show_cols = [col for col in ["title", "artist", "exclude_reason", "album", "release_date"] if col in excluded_preview_df.columns]
                st.dataframe(excluded_preview_df[show_cols], width="stretch")

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

    st.subheader("Analysis Metrics")
    debug_cols = st.columns(5)
    debug_cols[0].metric("Total songs in corpus", len(corpus_songs))
    debug_cols[1].metric("Songs with lyrics", len(songs_with_lyrics))
    debug_cols[2].metric("Total lyrics chars", total_lyrics_chars)
    debug_cols[3].metric("Raw words before cleaning", total_raw_words_before_clean)
    debug_cols[4].metric("Meaningful words after cleaning", int(analysis_result.summary["total_meaningful_tokens"]))

    summary = analysis_result.summary
    st.caption(f"Source used: {source_used}")
    st.caption(
        f"Filtering mode: {summary.get('filtering_strictness', filtering_strictness)} "
        f"(stopword version: {summary.get('stopword_version', STOPWORD_VERSION)})"
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Songs analyzed", int(summary["total_songs_analyzed"]))
    metric_cols[1].metric("Unique words", int(summary["unique_vocabulary_size"]))
    metric_cols[2].metric("Lexical diversity", f"{summary['lexical_diversity']:.3f}")
    metric_cols[3].metric("Meaningful tokens", int(summary["total_meaningful_tokens"]))

    st.subheader("Top words")
    st.plotly_chart(make_top_words_chart(analysis_result.word_frequencies_df), width="stretch")
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
                st.plotly_chart(cat_chart, width="stretch")
            st.json(categories_payload)

    st.subheader("Top bigrams")
    st.plotly_chart(make_top_bigrams_chart(analysis_result.bigram_frequencies_df), width="stretch")

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

    if show_wordcloud:
        image = make_wordcloud_image(analysis_result.word_frequencies_df)
        if image is not None:
            st.subheader("Word cloud")
            st.image(image, width="stretch")

    with st.expander("Included songs", expanded=False):
        included_df = pd.DataFrame(included_songs)
        if not included_df.empty:
            songs_to_show = st.selectbox(
                "Included songs rows to display",
                options=[10, 50, 100],
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
                    lambda n: "cache" if int(n) > 0 else "none"
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
        file_name="artist_wordprint_analysis.json",
        mime="application/json",
    )

if __name__ == "__main__":
    main()
