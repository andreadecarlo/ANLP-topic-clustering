"""BERTopic pipeline with human-readable topic labels for song lyrics."""

from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from anlp.config import (
    BERTOPIC_EMBEDDING_MODEL,
    BERTOPIC_MIN_TOPIC_SIZE,
    BERTOPIC_NUM_TOPICS,
    BERTOPIC_REPRESENTATION_N_WORDS,
    BERTOPIC_REPRESENTATION_NGRAM_RANGE,
    MODELS_DIR,
    ensure_dirs,
)
from anlp.data.load_lyrics import load_lyrics_subset, get_lyrics_corpus


def build_bertopic_model(
    documents: list[str],
    embedding_model: str = BERTOPIC_EMBEDDING_MODEL,
    min_topic_size: int = BERTOPIC_MIN_TOPIC_SIZE,
    nr_topics: str | int = BERTOPIC_NUM_TOPICS,
    n_gram_range: tuple[int, int] = BERTOPIC_REPRESENTATION_NGRAM_RANGE,
    n_words: int = BERTOPIC_REPRESENTATION_N_WORDS,
) -> BERTopic:
    """
    Build and fit BERTopic with human-readable topic representation.
    Uses n-gram range (1, 2) and top words for interpretable labels.
    """
    # Vectorizer for c-TF-IDF topic representation (phrases + single words)
    vectorizer = CountVectorizer(
        ngram_range=n_gram_range,
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        vectorizer_model=vectorizer,
        verbose=True,
    )

    topic_model.fit(documents)
    # Refine topic representation for human-readable labels (phrases)
    topic_model.update_topics(
        documents,
        n_gram_range=n_gram_range,
        n_words=n_words,
    )
    return topic_model


def get_topic_labels(topic_model: BERTopic) -> dict[int, str]:
    """Get human-readable topic labels (top terms or custom labels)."""
    info = topic_model.get_topic_info()
    labels = {}
    for _, row in info.iterrows():
        tid = row["Topic"]
        name = row["Name"]
        if tid == -1:
            labels[-1] = "Outliers"
        else:
            # Name is already "topic_0_word1_word2_..." – make it readable
            label = name.replace("_", " ").strip()
            if label.startswith("topic "):
                label = " ".join(label.split()[2:])  # drop "topic N"
            labels[tid] = label or str(tid)
    return labels


def fit_bertopic_on_lyrics(
    year_min: int = 2010,
    year_max: int = 2020,
    max_docs: int | None = 50_000,
    save_path: Path | None = None,
) -> tuple[BERTopic, pd.DataFrame, list[int], list[float]]:
    """
    Load lyrics subset, fit BERTopic, return model, metadata DataFrame, topics, and probs.
    """
    ensure_dirs()
    df = load_lyrics_subset(year_min=year_min, year_max=year_max, max_docs=max_docs)
    corpus, df_clean = get_lyrics_corpus(df, min_chars=100)

    topic_model = build_bertopic_model(corpus)
    topics, probs = topic_model.transform(corpus)

    df_clean = df_clean.copy()
    df_clean["topic"] = topics
    df_clean["topic_prob"] = probs

    if save_path is None:
        save_path = MODELS_DIR / "bertopic_lyrics"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(save_path))
    df_clean.to_parquet(save_path.parent / (save_path.name + "_docs.parquet"), index=False)

    return topic_model, df_clean, topics, probs


def load_bertopic_model(model_path: Path | str) -> BERTopic:
    """Load a saved BERTopic model."""
    return BERTopic.load(str(model_path))
