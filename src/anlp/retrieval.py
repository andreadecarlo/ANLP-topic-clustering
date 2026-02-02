"""Retrieval: similar songs by topic, representative songs per topic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from anlp.config import TOP_K_SIMILAR_SONGS, TOP_K_REPRESENTATIVE_SONGS


def similar_songs_for_song(
    doc_id: int,
    docs_df: pd.DataFrame,
    topics: list[int],
    probs: list[float] | np.ndarray | None = None,
    top_k: int = TOP_K_SIMILAR_SONGS,
    title_col: str = "title",
    artist_col: str = "artist",
) -> pd.DataFrame:
    """
    Given a document index (song), return the most similar songs based on topic.
    Similarity = same topic, then by topic probability (if probs available).
    """
    if doc_id < 0 or doc_id >= len(topics):
        raise ValueError(f"doc_id must be in [0, {len(topics)-1}]")
    topic = topics[doc_id]
    if topic == -1:
        # Outlier: no topic; could use embedding similarity if we stored embeddings
        return docs_df.iloc[[]].copy()

    same_topic_mask = np.array(topics) == topic
    indices = np.where(same_topic_mask)[0]
    if probs is not None:
        probs_arr = np.asarray(probs)
        if probs_arr.ndim == 2:
            topic_probs = probs_arr[indices, topic] if topic < probs_arr.shape[1] else np.ones(len(indices))
        else:
            topic_probs = np.array(probs)[indices]
        order = np.argsort(-topic_probs)
        indices = indices[order]

    # Exclude the query song
    indices = indices[indices != doc_id][: top_k + 1]
    if len(indices) > top_k:
        indices = indices[:top_k]

    out = docs_df.iloc[indices].copy()
    if "topic" not in out.columns and len(docs_df.columns) <= len(docs_df):
        out["topic"] = topic
    cols = [c for c in [title_col, artist_col, "topic", "year"] if c in out.columns]
    return out[cols] if cols else out


def representative_songs_for_topic(
    topic_id: int,
    docs_df: pd.DataFrame,
    topics: list[int],
    probs: list[float] | np.ndarray | None = None,
    top_k: int = TOP_K_REPRESENTATIVE_SONGS,
    title_col: str = "title",
    artist_col: str = "artist",
) -> pd.DataFrame:
    """
    Given a topic ID, return the most representative songs (highest topic probability).
    """
    same_topic_mask = np.array(topics) == topic_id
    indices = np.where(same_topic_mask)[0]
    if len(indices) == 0:
        return docs_df.iloc[[]].copy()

    if probs is not None:
        probs_arr = np.asarray(probs)
        if probs_arr.ndim == 2 and topic_id < probs_arr.shape[1]:
            topic_probs = probs_arr[indices, topic_id]
        else:
            topic_probs = np.ones(len(indices))
        order = np.argsort(-topic_probs)
        indices = indices[order]

    indices = indices[:top_k]
    out = docs_df.iloc[indices].copy()
    cols = [c for c in [title_col, artist_col, "topic", "year"] if c in out.columns]
    return out[cols] if cols else out


def similar_songs_for_song_from_model(
    topic_model: "BERTopic",
    docs_df: pd.DataFrame,
    doc_id: int,
    corpus: list[str],
    top_k: int = TOP_K_SIMILAR_SONGS,
) -> pd.DataFrame:
    """
    Use BERTopic's topic distribution: get topic for doc_id, then representative docs
    for that topic (or same-topic docs from fit result).
    """
    if hasattr(topic_model, "topics_") and topic_model.topics_ is not None:
        topics = topic_model.topics_
        probs = getattr(topic_model, "probabilities_", None)
        return similar_songs_for_song(doc_id, docs_df, topics, probs, top_k=top_k)

    # Fallback: transform the single document and find topic, then get representative
    doc = corpus[doc_id]
    topic, prob = topic_model.transform([doc])
    topic = topic[0]
    if topic == -1:
        return docs_df.iloc[[]].copy()
    # Get all docs from the same topic (we need topics for full corpus)
    if topic_model.topics_ is not None:
        return representative_songs_for_topic(
            topic, docs_df, topic_model.topics_, topic_model.probabilities_, top_k=top_k
        )
    return docs_df.iloc[[]].copy()
