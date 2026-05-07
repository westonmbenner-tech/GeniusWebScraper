"""Typed data models for Lyric Atlas v2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Artist:
    id: str
    name: str
    slug: str
    image_url: str | None = None
    bio_summary: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class Song:
    id: str
    artist_id: str
    title: str
    slug: str
    r2_lyrics_key: str
    album: str | None = None
    release_year: int | None = None
    track_number: int | None = None
    source_url: str | None = None
    word_count: int | None = None
    unique_word_count: int | None = None
    lexical_diversity: float | None = None


@dataclass
class AnalysisRun:
    id: str
    artist_id: str
    run_id: str
    status: str
    model_version: str | None = None
    algorithm_version: str | None = None
    song_count: int | None = None
    total_words: int | None = None
    r2_raw_corpus_key: str | None = None
    r2_full_analysis_key: str | None = None
    r2_per_song_analysis_key: str | None = None
    r2_debug_key: str | None = None
    error_message: str | None = None


@dataclass
class WordStat:
    id: str
    artist_id: str
    analysis_run_id: str
    word: str
    count: int
    rank: int
    frequency_per_1000: float | None = None
    part_of_speech: str | None = None


@dataclass
class PhraseStat:
    id: str
    artist_id: str
    analysis_run_id: str
    phrase: str
    count: int
    rank: int
    phrase_length: int | None = None


@dataclass
class ThemeStat:
    id: str
    artist_id: str
    analysis_run_id: str
    theme: str
    score: float | None = None
    evidence_count: int | None = None
    rank: int | None = None


@dataclass
class ArtistProfile:
    id: str
    artist_id: str
    analysis_run_id: str
    summary: str | None = None
    style_notes: str | None = None
    common_themes: list[str] | None = None
    signature_words: list[str] | None = None
    signature_phrases: list[str] | None = None
    emotional_palette: list[str] | None = None


@dataclass
class SongAnalysisSummary:
    id: str
    song_id: str
    analysis_run_id: str
    short_summary: str | None = None
    dominant_themes: list[str] | None = None
    mood: str | None = None
    notable_words: list[str] | None = None
    notable_phrases: list[str] | None = None
    r2_detailed_analysis_key: str | None = None
