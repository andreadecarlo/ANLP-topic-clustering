"""Compare topic modeling algorithms with OCTIS on lyrics subset."""

from pathlib import Path

import pandas as pd

from anlp.config import (
    OCTIS_ALGORITHMS,
    OCTIS_NUM_TOPICS,
    OCTIS_RANDOM_STATE,
    MODELS_DIR,
    ensure_dirs,
)
from anlp.data.load_lyrics import load_lyrics_subset, get_lyrics_corpus, tokenize_for_octis


def build_octis_dataset(tokenized_corpus: list[list[str]]) -> tuple["Dataset", list[list[str]]]:
    """Build OCTIS Dataset from list of tokenized documents (list of list of words).
    Returns (Dataset, tokenized_corpus) so we can use tokenized_corpus for coherence metrics.
    """
    from octis.dataset.dataset import Dataset

    # Build vocabulary (unique words, preserve order)
    vocab_set: dict[str, None] = {}
    for doc in tokenized_corpus:
        for w in doc:
            vocab_set[w] = None
    vocabulary = list(vocab_set.keys())

    # Filter very short docs (OCTIS needs enough tokens); keep aligned tokenized for coherence
    tokenized_filtered = [doc for doc in tokenized_corpus if len(doc) >= 5]

    # OCTIS LDA expects metadata with "last-training-doc" for get_partitioned_corpus();
    # use full corpus as training (no separate test split) so it returns (train, test) unpackable.
    # LDA expects corpus as list of lists of word strings (not indices), so pass tokenized_filtered directly.
    n = len(tokenized_filtered)
    metadata = {"last-training-doc": n}

    return Dataset(corpus=tokenized_filtered, vocabulary=vocabulary, metadata=metadata), tokenized_filtered


def run_octis_model(
    dataset: "Dataset",
    algorithm: str,
    num_topics: int = OCTIS_NUM_TOPICS,
    random_state: int = OCTIS_RANDOM_STATE,
) -> dict:
    """Train one OCTIS model and return output (topics, topic-document-matrix, etc.)."""
    if algorithm.upper() == "LDA":
        from octis.models.LDA import LDA

        model = LDA(
            num_topics=num_topics,
            alpha="symmetric",
            eta=0.01,
            iterations=50,
            random_state=random_state,
        )
        output = model.train_model(dataset, hyperparams={}, top_words=15)
    elif algorithm.upper() == "NMF":
        from octis.models.NMF import NMF

        model = NMF(num_topics=num_topics, random_state=random_state)
        output = model.train_model(dataset, hyperparameters={}, top_words=15)
    elif algorithm.upper() == "LSI":
        from octis.models.LSI import LSI

        model = LSI(num_topics=num_topics, random_state=random_state)
        output = model.train_model(dataset, hyperparameters={}, top_words=15)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {OCTIS_ALGORITHMS}")

    return output


def evaluate_octis(
    output: dict,
    tokenized_corpus: list[list[str]] | None = None,
) -> dict:
    """Compute OCTIS metrics (coherence, diversity) for model output."""
    from octis.evaluation_metrics.coherence_metrics import Coherence
    from octis.evaluation_metrics.diversity_metrics import TopicDiversity

    diversity = TopicDiversity(topk=10)
    diversity_score = diversity.score(output)

    coherence_score = None
    if tokenized_corpus:
        coherence = Coherence(texts=tokenized_corpus, topk=10, measure="c_npmi")
        coherence_score = coherence.score(output)

    return {
        "coherence_npmi": coherence_score,
        "topic_diversity": diversity_score,
    }


def compare_octis(
    year_min: int = 2010,
    year_max: int = 2020,
    max_docs: int | None = 50_000,
    algorithms: list[str] | None = None,
    num_topics: int = OCTIS_NUM_TOPICS,
) -> pd.DataFrame:
    """
    Load lyrics subset, build OCTIS dataset, run and evaluate each algorithm.
    Returns a DataFrame with algorithm names and metrics.
    """
    ensure_dirs()
    algorithms = algorithms or OCTIS_ALGORITHMS

    df = load_lyrics_subset(year_min=year_min, year_max=year_max, max_docs=max_docs)
    corpus, df_clean = get_lyrics_corpus(df, min_chars=100)
    tokenized = tokenize_for_octis(corpus)
    dataset, tokenized_filtered = build_octis_dataset(tokenized)

    results = []
    for alg in algorithms:
        output = run_octis_model(dataset, alg, num_topics=num_topics)
        metrics = evaluate_octis(output, tokenized_corpus=tokenized_filtered)
        results.append({"algorithm": alg, **metrics})

    return pd.DataFrame(results)
