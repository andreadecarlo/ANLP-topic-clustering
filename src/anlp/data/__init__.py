"""Data loading and preparation for lyrics corpus."""

from anlp.data.load_lyrics import (
    get_lyrics_corpus,
    load_lyrics_subset,
    tokenize_for_octis,
)

__all__ = ["load_lyrics_subset", "get_lyrics_corpus", "tokenize_for_octis"]
