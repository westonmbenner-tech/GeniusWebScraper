"""Helpers for multi-artist lexical comparison."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def normalize_song_dict(song: dict[str, Any]) -> dict[str, Any]:
    """Ensure lyrics and lyrics_char_count are consistent for analysis."""
    d = dict(song)
    lyrics = str(d.get("lyrics", "") or "")
    d["lyrics"] = lyrics
    body_len = len(lyrics)
    d["lyrics_char_count"] = body_len
    return d


def shared_top100_rank_table(word_freq_by_artist: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Words that appear in every artist's top-100 list, with rank (1 = most frequent) and count per artist.
    """
    if len(word_freq_by_artist) < 2:
        return pd.DataFrame()

    labels = list(word_freq_by_artist.keys())
    rank_maps: dict[str, dict[str, int]] = {}
    count_maps: dict[str, dict[str, int]] = {}
    top_sets: list[set[str]] = []

    for label, df in word_freq_by_artist.items():
        sub = df.head(100)
        rmap: dict[str, int] = {}
        cmap: dict[str, int] = {}
        for i, (_, row) in enumerate(sub.iterrows()):
            w = str(row["word"])
            rmap[w] = i + 1
            cmap[w] = int(row["count"])
        rank_maps[label] = rmap
        count_maps[label] = cmap
        top_sets.append(set(rmap.keys()))

    shared = set.intersection(*top_sets)
    if not shared:
        return pd.DataFrame(columns=["word"] + [f"{l} rank" for l in labels] + [f"{l} count" for l in labels])

    def avg_rank(w: str) -> float:
        return sum(rank_maps[l][w] for l in labels) / len(labels)

    rows: list[dict[str, Any]] = []
    for w in sorted(shared, key=avg_rank):
        row: dict[str, Any] = {"word": w}
        for label in labels:
            row[f"{label} rank"] = rank_maps[label][w]
            row[f"{label} count"] = count_maps[label][w]
        rows.append(row)

    return pd.DataFrame(rows)
