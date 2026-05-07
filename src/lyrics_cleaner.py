"""Lyrics cleaning and token normalization utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


CUSTOM_STOPWORDS = {
    "yeah",
    "oh",
    "ooh",
    "ah",
    "uh",
    "hey",
    "la",
    "na",
    "woah",
    "whoa",
    "baby",
    "chorus",
    "verse",
    "intro",
    "outro",
    "bridge",
    "refrain",
    "lyrics",
    "embed",
    "contributor",
    "contributors",
    "eh",
    "uhh",
    "umm",
    "um",
    "like",
    "hmm",
    "mmm",
    "mm",
    "gon",
    "gonna",
    "wanna",
    "gotta",
    "yuh",
    "ayy",
    "nah",
    "ya",
    "yo",
}


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if missing."""
    for resource in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def remove_genius_artifacts(text: str) -> str:
    """Strip common non-lyric Genius artifacts."""
    cleaned = re.sub(r"\[.*?\]", " ", text)
    cleaned = re.sub(r"\bLyrics\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bEmbed\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"You might also like", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\d+ Contributors?.*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


@dataclass
class SongTokenResult:
    tokens: list[str]
    total_tokens_before_filtering: int


def clean_lyrics_tokens(lyrics_text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> SongTokenResult:
    """Clean and tokenize lyrics into meaningful lemmatized tokens."""
    if not lyrics_text:
        return SongTokenResult(tokens=[], total_tokens_before_filtering=0)

    stripped = remove_genius_artifacts(lyrics_text).lower()

    try:
        raw_tokens = word_tokenize(stripped)
    except LookupError:
        raw_tokens = re.findall(r"[a-zA-Z']+", stripped)

    total_tokens = len(raw_tokens)
    cleaned_tokens: list[str] = []

    for token in raw_tokens:
        token = re.sub(r"[^a-z']", "", token)
        if not token:
            continue
        if len(token) == 1 and token not in {"i", "a"}:
            continue
        if token.isdigit():
            continue
        if token in stop_words:
            continue
        lemma = lemmatizer.lemmatize(token)
        if lemma in stop_words or not lemma:
            continue
        cleaned_tokens.append(lemma)

    return SongTokenResult(tokens=cleaned_tokens, total_tokens_before_filtering=total_tokens)


def build_stop_words() -> set[str]:
    """Build combined standard and custom stopword set."""
    ensure_nltk_resources()
    english = set(stopwords.words("english"))
    return english.union(CUSTOM_STOPWORDS)


def clean_song_lyrics(song_lyrics: Iterable[str]) -> tuple[list[list[str]], list[int]]:
    """Clean all songs and return per-song tokens and pre-filter token counts."""
    stop_words = build_stop_words()
    lemmatizer = WordNetLemmatizer()
    per_song_tokens: list[list[str]] = []
    totals_before: list[int] = []

    for lyrics in song_lyrics:
        result = clean_lyrics_tokens(lyrics or "", stop_words, lemmatizer)
        per_song_tokens.append(result.tokens)
        totals_before.append(result.total_tokens_before_filtering)

    return per_song_tokens, totals_before
