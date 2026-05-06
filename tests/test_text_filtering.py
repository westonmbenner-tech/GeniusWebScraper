from src.text_filtering import tokenize_and_filter_lyrics


def test_contractions_expand_and_original_forms_removed() -> None:
    result = tokenize_and_filter_lyrics("there's no way I can't go 'cause I won't", strictness="lyric_clean")
    assert "there's" not in result.raw_tokens
    assert "can't" not in result.raw_tokens
    assert "'cause" not in result.raw_tokens
    assert "there" not in result.meaningful_tokens
    assert "is" not in result.meaningful_tokens
    assert "can" not in result.meaningful_tokens
    assert "not" not in result.meaningful_tokens


def test_genius_artifacts_removed() -> None:
    result = tokenize_and_filter_lyrics("123Embed [Chorus] Home road love", strictness="lyric_clean")
    assert "embed" not in result.normalized_tokens
    assert "chorus" not in result.normalized_tokens
    assert "home" in result.meaningful_tokens
    assert "road" in result.meaningful_tokens
    assert "love" in result.meaningful_tokens


def test_theme_focused_removes_maybe_words() -> None:
    result = tokenize_and_filter_lyrics("time way home", strictness="theme_focused")
    assert "time" not in result.meaningful_tokens
    assert "way" not in result.meaningful_tokens
    assert "home" in result.meaningful_tokens


def test_lyric_clean_keeps_maybe_words_and_theme_words() -> None:
    result = tokenize_and_filter_lyrics(
        "time way home cold hell love road god",
        strictness="lyric_clean",
    )
    assert "time" in result.meaningful_tokens
    assert "way" in result.meaningful_tokens
    for token in ["home", "cold", "hell", "love", "road", "god"]:
        assert token in result.meaningful_tokens
