"""Vocabulary analytics for cleaned lyric tokens."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class AnalysisResult:
    summary: dict[str, Any]
    word_frequencies_df: pd.DataFrame
    bigram_frequencies_df: pd.DataFrame
    song_stats_df: pd.DataFrame
    top_words: list[dict[str, Any]]


def analyze_tokens(
    songs: list[dict[str, Any]],
    per_song_tokens: list[list[str]],
    prefilter_counts: list[int],
) -> AnalysisResult:
    """Compute aggregate and per-song lexical statistics."""
    all_tokens = [token for song_tokens in per_song_tokens for token in song_tokens]
    word_counter = Counter(all_tokens)

    bigrams: list[str] = []
    for song_tokens in per_song_tokens:
        bigrams.extend([f"{song_tokens[i]} {song_tokens[i + 1]}" for i in range(len(song_tokens) - 1)])
    bigram_counter = Counter(bigrams)

    word_df = pd.DataFrame(word_counter.most_common(100), columns=["word", "count"])
    bigram_df = pd.DataFrame(bigram_counter.most_common(50), columns=["bigram", "count"])

    song_rows = []
    for idx, song in enumerate(songs):
        tokens = per_song_tokens[idx] if idx < len(per_song_tokens) else []
        total_before = prefilter_counts[idx] if idx < len(prefilter_counts) else 0
        song_rows.append(
            {
                "title": song.get("title"),
                "artist": song.get("artist"),
                "genius_song_id": song.get("genius_song_id"),
                "url": song.get("url"),
                "album": song.get("album"),
                "release_date": song.get("release_date"),
                "tokens_before_filtering": total_before,
                "meaningful_tokens": len(tokens),
                "unique_meaningful_tokens": len(set(tokens)),
            }
        )

    song_stats_df = pd.DataFrame(song_rows)
    total_meaningful = len(all_tokens)
    unique_vocab = len(word_counter)
    lexical_diversity = (unique_vocab / total_meaningful) if total_meaningful else 0.0

    summary: dict[str, float | int] = {
        "total_songs_analyzed": len(songs),
        "total_tokens_before_filtering": int(sum(prefilter_counts)),
        "total_meaningful_tokens": total_meaningful,
        "unique_vocabulary_size": unique_vocab,
        "lexical_diversity": lexical_diversity,
    }

    top_words = [{"word": row["word"], "count": int(row["count"])} for _, row in word_df.iterrows()]

    return AnalysisResult(
        summary=summary,
        word_frequencies_df=word_df,
        bigram_frequencies_df=bigram_df,
        song_stats_df=song_stats_df,
        top_words=top_words,
    )
