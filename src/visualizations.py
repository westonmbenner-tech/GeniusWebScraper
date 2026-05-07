"""Plotly and word cloud visualization helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
import plotly.express as px
from wordcloud import WordCloud


def make_top_words_chart(word_df: pd.DataFrame):
    """Create bar chart of top words."""
    top = word_df.head(25)
    if top.empty:
        fig = px.bar(title="Top 25 Meaningful Words")
        fig.update_layout(xaxis_visible=False, yaxis_visible=False)
        return fig
    return px.bar(top, x="word", y="count", title="Top 25 Meaningful Words")


def make_top_bigrams_chart(bigram_df: pd.DataFrame):
    """Create bar chart of top bigrams."""
    top = bigram_df.head(20)
    if top.empty:
        fig = px.bar(title="Top 20 Bigrams")
        fig.update_layout(xaxis_visible=False, yaxis_visible=False)
        return fig
    return px.bar(top, x="bigram", y="count", title="Top 20 Bigrams")


def make_comparison_top25_grouped_bar(word_dfs: dict[str, pd.DataFrame]):
    """Grouped bar chart: each artist's top-25 words on a shared x-axis (union of top-25 words)."""
    if not word_dfs:
        return None
    word_totals: Counter[str] = Counter()
    top25_maps: dict[str, dict[str, int]] = {}
    for label, df in word_dfs.items():
        sub = df.head(25)
        top25_maps[label] = {str(row["word"]): int(row["count"]) for _, row in sub.iterrows()}
        for _, row in sub.iterrows():
            word_totals[str(row["word"])] += int(row["count"])
    ordered_words = [w for w, _ in word_totals.most_common()]
    rows: list[dict[str, Any]] = []
    for label in word_dfs:
        m = top25_maps[label]
        for w in ordered_words:
            rows.append({"artist": label, "word": w, "count": int(m.get(w, 0))})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return None
    fig = px.bar(
        plot_df,
        x="word",
        y="count",
        color="artist",
        barmode="group",
        title="Top 25 words per artist (grouped counts)",
        category_orders={"word": ordered_words},
    )
    fig.update_layout(xaxis_tickangle=-45, height=620, legend_title_text="Artist")
    return fig


def make_category_chart(categories_payload: dict[str, Any]):
    """Create category total-count chart from OpenAI payload."""
    rows = []
    for category in categories_payload.get("categories", []):
        count_sum = sum(word["count"] for word in category.get("words", []))
        rows.append({"category": category.get("name"), "count": count_sum})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    return px.bar(df, x="category", y="count", title="Category Totals")


def make_wordcloud_image(word_df: pd.DataFrame, *, top_n: int = 25):
    """Generate optional word cloud from the highest-frequency words (default: top 25, like the bar chart)."""
    if word_df.empty or top_n <= 0:
        return None
    sub = word_df.head(int(top_n))
    freqs = {row["word"]: int(row["count"]) for _, row in sub.iterrows()}
    if not freqs:
        return None
    wc = WordCloud(width=1200, height=600, background_color="white").generate_from_frequencies(freqs)
    return wc.to_array()
