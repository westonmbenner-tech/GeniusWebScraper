"""Optional semantic word categorization with OpenAI."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


def _validate_payload(payload: dict[str, Any]) -> bool:
    """Validate expected strict JSON category payload."""
    categories = payload.get("categories")
    if not isinstance(categories, list):
        return False
    for category in categories:
        if not isinstance(category, dict):
            return False
        if not isinstance(category.get("name"), str):
            return False
        if not isinstance(category.get("description"), str):
            return False
        words = category.get("words")
        if not isinstance(words, list):
            return False
        for item in words:
            if not isinstance(item, dict):
                return False
            if not isinstance(item.get("word"), str):
                return False
            if not isinstance(item.get("count"), int):
                return False
    return True


def categorize_top_words(top_words: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    """Group top words into semantic categories using OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY is missing. Skipping categorization."

    if not top_words:
        return None, "No words available to categorize."

    client = OpenAI(api_key=api_key)
    prompt = (
        "Group these artist vocabulary words into 6-10 semantic categories. "
        "Return strict JSON only using this schema: "
        '{"categories":[{"name":"category name","description":"short explanation","words":[{"word":"word","count":123}]}]}. '
        "Do not include markdown. Input words:\n"
        f"{json.dumps(top_words, ensure_ascii=False)}"
    )

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": "You are a strict JSON generator. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.output_text
        parsed = json.loads(content)
        if not _validate_payload(parsed):
            return None, "OpenAI returned invalid category JSON."
        return parsed, None
    except Exception as exc:  # noqa: BLE001
        return None, f"OpenAI categorization failed: {exc}"
