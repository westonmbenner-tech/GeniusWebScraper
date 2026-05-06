"""Text filtering and tokenization utilities for lyrics analysis."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import nltk
from nltk.corpus import stopwords

STOPWORD_VERSION = "v1.0"

CONTRACTION_MAP = {
    "there's": "there is",
    "that's": "that is",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "it's": "it is",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would",
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "ain't": "is not",
    "'cause": "because",
    "cause": "because",
    "'til": "until",
    "til": "until",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "lemme": "let me",
    "gimme": "give me",
    "tryna": "trying to",
    "outta": "out of",
    "lotta": "lot of",
    "coulda": "could have",
    "shoulda": "should have",
    "woulda": "would have",
}

DEFINITE_LYRIC_STOPWORDS = {
    "yeah", "yea", "yep", "nope",
    "oh", "ooh", "oooh", "ah", "uh", "um", "hmm",
    "hey", "ha", "haha",
    "la", "na", "da", "doo",
    "woah", "whoa", "mmm",
    "chorus", "verse", "intro", "outro", "bridge", "refrain",
    "pre", "hook",
    "lyrics", "lyric", "embed", "contributors", "contributor",
    "know", "go", "well", "never", "could", "get", "take",
    "every", "one", "make", "still", "say", "let", "thing",
    "need", "come", "would", "think", "always", "much", "ever",
    "used", "got", "even", "something", "everything", "rather",
    "around",
}

MAYBE_LYRIC_STOPWORDS = {
    "time", "see", "way", "hold", "feel", "day", "away",
    "back", "right", "mind", "head", "night", "world",
    "maybe", "long", "face", "life", "hand", "keep", "stand",
    "find", "good", "told", "old", "eye", "close", "hard", "live",
    "run", "left", "thought",
}

MEANINGFUL_THEME_WORDS = {
    "home", "man", "hope", "love", "far", "cold", "hell",
    "burn", "damn", "lost", "alone", "dead", "hate", "road",
    "pain", "dear", "leave", "lord", "son", "broken", "die",
    "dark", "better", "late", "fire", "fear", "tired", "heart",
    "word", "god", "gone", "tonight", "light", "soul", "wall",
}

StrictnessMode = Literal["basic", "lyric_clean", "theme_focused"]

_CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTION_MAP, key=len, reverse=True)) + r")\b"
)


@dataclass
class TokenizationResult:
    raw_tokens: list[str]
    normalized_tokens: list[str]
    meaningful_tokens: list[str]
    removed_tokens: list[dict[str, str]]
    counts_by_reason: dict[str, int]


def ensure_nltk_resources() -> None:
    """Download stopwords corpus if missing."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def normalize_contractions(text: str) -> str:
    """Normalize apostrophes and expand contractions safely."""
    lowered = _normalize_quotes(text).lower()
    return _CONTRACTION_PATTERN.sub(lambda m: CONTRACTION_MAP[m.group(0)], lowered)


def _remove_genius_artifacts(text: str) -> str:
    cleaned = re.sub(r"\[.*?\]", " ", text)
    cleaned = re.sub(r"\b(?:lyrics|embed|you might also like)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\d+\s*contributors?.*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\d+\s*embed\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def get_standard_stopwords() -> set[str]:
    ensure_nltk_resources()
    base = set(stopwords.words("english"))
    return base.difference(MEANINGFUL_THEME_WORDS)


def get_stopword_sets(strictness: StrictnessMode) -> tuple[set[str], set[str], set[str]]:
    standard = get_standard_stopwords()
    definite = DEFINITE_LYRIC_STOPWORDS.difference(MEANINGFUL_THEME_WORDS)
    maybe = MAYBE_LYRIC_STOPWORDS.difference(MEANINGFUL_THEME_WORDS)
    if strictness == "basic":
        return standard, {"chorus", "verse", "intro", "outro", "bridge", "refrain", "embed", "lyrics"}, set()
    if strictness == "theme_focused":
        return standard, definite, maybe
    return standard, definite, set()


def tokenize_and_filter_lyrics(lyrics: str, strictness: StrictnessMode) -> TokenizationResult:
    """Tokenize lyrics and return detailed filtering diagnostics."""
    if not lyrics:
        return TokenizationResult([], [], [], [], {})

    cleaned = _remove_genius_artifacts(lyrics)
    normalized = normalize_contractions(cleaned)
    no_punct = re.sub(r"[^a-z\s]", " ", normalized)
    raw_tokens = normalized.split()
    normalized_tokens = no_punct.split()

    standard_stopwords, definite_stopwords, maybe_stopwords = get_stopword_sets(strictness)
    meaningful_tokens: list[str] = []
    removed_tokens: list[dict[str, str]] = []
    counts = Counter()

    for token in normalized_tokens:
        if not token.isalpha():
            reason = "non_alpha"
        elif len(token) <= 1:
            reason = "too_short"
        elif token in standard_stopwords:
            reason = "standard_stopword"
        elif token in definite_stopwords:
            reason = "definite_lyric_stopword"
        elif token in maybe_stopwords:
            reason = "maybe_lyric_stopword"
        else:
            reason = ""

        if reason:
            removed_tokens.append({"token": token, "reason": reason})
            counts[reason] += 1
            continue
        meaningful_tokens.append(token)

    return TokenizationResult(
        raw_tokens=raw_tokens,
        normalized_tokens=normalized_tokens,
        meaningful_tokens=meaningful_tokens,
        removed_tokens=removed_tokens,
        counts_by_reason=dict(counts),
    )


def aggregate_removed_tokens(tokenization_results: list[TokenizationResult]) -> list[dict[str, str | int]]:
    """Aggregate removed tokens across songs with reason for diagnostics display."""
    counter: Counter[tuple[str, str]] = Counter()
    for result in tokenization_results:
        for removed in result.removed_tokens:
            token = str(removed.get("token", ""))
            reason = str(removed.get("reason", ""))
            if token and reason:
                counter[(token, reason)] += 1

    rows: list[dict[str, str | int]] = []
    for (token, reason), count in counter.most_common(50):
        rows.append({"token": token, "count": int(count), "reason": reason})
    return rows
