"""Compare topic modeling algorithms with OCTIS on lyrics subset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from bertopic import BERTopic

from anlp.config import (
    MAX_DOCS_SUBSET,
    MODELS_DIR,
    OCTIS_ALGORITHMS,
    OCTIS_NUM_TOPICS,
    OCTIS_RANDOM_STATE,
    ensure_dirs,
)
from anlp.data.load_lyrics import load_lyrics_subset, get_lyrics_corpus, tokenize_for_octis


def bertopic_to_octis_output(topic_model: BERTopic, topk: int = 10) -> dict:
    """Convert a fitted BERTopic model to OCTIS-style output for coherence/diversity.
    Returns dict with 'topics': list of list of top words (skips outlier topic -1).
    """
    raw_topics = topic_model.get_topics()
    topics = []
    for tid in sorted(raw_topics.keys()):
        if tid == -1:
            continue
        words = raw_topics[tid]
        top_words = [w for w, _ in (words or [])[:topk]]
        if top_words:
            topics.append(top_words)
    return {"topics": topics}


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
    elif algorithm.upper() == "CTM":
        # NumPy 2 removed np.Inf; OCTIS early_stopping still uses it
        import numpy as np

        if not hasattr(np, "Inf"):
            np.Inf = np.inf  # type: ignore[attr-defined]

        from octis.models.CTM import CTM

        # use_partitions=False: we have no train/val/test split; bert_path caches BERT embeddings
        corpus_size = len(dataset.get_corpus()) or 64
        # Include corpus size in path so cache is not reused across different doc counts (avoids BoW/embedding size mismatch)
        ctm_bert_dir = MODELS_DIR / "ctm_bert"
        ctm_bert_dir.mkdir(parents=True, exist_ok=True)
        bert_path = str(ctm_bert_dir / f"n{corpus_size}")
        model = CTM(
            num_topics=num_topics,
            num_epochs=50,
            batch_size=min(64, max(8, corpus_size)),
            use_partitions=False,
            bert_path=bert_path,
        )
        output = model.train_model(dataset, hyperparameters={}, top_words=15)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {OCTIS_ALGORITHMS}")

    return output


def evaluate_octis(
    output: dict,
    tokenized_corpus: list[list[str]] | None = None,
    topk: int = 10,
) -> dict:
    """Compute OCTIS metrics (coherence, diversity) for model output.
    If topics have fewer than topk words (e.g. small corpus/BERTopic), topk is reduced.
    """
    from octis.evaluation_metrics.coherence_metrics import Coherence
    from octis.evaluation_metrics.diversity_metrics import TopicDiversity

    topics = output.get("topics") or []
    min_words = min((len(t) for t in topics), default=topk)
    effective_topk = max(1, min(topk, min_words))

    diversity = TopicDiversity(topk=effective_topk)
    diversity_score = diversity.score(output)

    coherence_score = None
    if tokenized_corpus:
        coherence = Coherence(
            texts=tokenized_corpus, topk=effective_topk, measure="c_npmi"
        )
        coherence_score = coherence.score(output)

    return {
        "coherence_npmi": coherence_score,
        "topic_diversity": diversity_score,
    }


def compare_octis(
    year_min: int = 2010,
    year_max: int = 2020,
    max_docs: int | None = MAX_DOCS_SUBSET,
    algorithms: list[str] | None = None,
    num_topics: int = OCTIS_NUM_TOPICS,
    include_bertopic: bool = False,
    bertopic_model_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load lyrics subset, build OCTIS dataset, run and evaluate each algorithm.
    If include_bertopic=True, evaluate BERTopic with the same coherence/diversity metrics.
    If bertopic_model_path is set, load that saved model; otherwise fit BERTopic on the
    same corpus (same doc filter as OCTIS).
    Returns a DataFrame with algorithm names and metrics.
    """
    ensure_dirs()
    algorithms = algorithms or OCTIS_ALGORITHMS

    dataset = load_lyrics_subset(
        year_min=year_min, year_max=year_max, max_docs=max_docs
    )
    corpus, _ = get_lyrics_corpus(dataset, min_chars=100)
    tokenized = tokenize_for_octis(corpus)
    dataset, tokenized_filtered = build_octis_dataset(tokenized)

    results = []
    for alg in algorithms:
        output = run_octis_model(dataset, alg, num_topics=num_topics)
        metrics = evaluate_octis(output, tokenized_corpus=tokenized_filtered)
        results.append({"algorithm": alg, **metrics})

    if include_bertopic:
        from anlp.bertopic_pipeline import build_bertopic_model, load_bertopic_model

        if bertopic_model_path is not None and Path(bertopic_model_path).exists():
            topic_model = load_bertopic_model(bertopic_model_path)
        else:
            # Same docs as OCTIS: raw text of docs that passed len(doc) >= 5
            corpus_for_bertopic = [
                corpus[i] for i, doc in enumerate(tokenized) if len(doc) >= 5
            ]
            topic_model = build_bertopic_model(
                corpus_for_bertopic,
                nr_topics=num_topics,
            )
        bertopic_output = bertopic_to_octis_output(topic_model, topk=10)
        metrics = evaluate_octis(
            bertopic_output, tokenized_corpus=tokenized_filtered
        )
        results.append({"algorithm": "BERTopic", **metrics})

    return pd.DataFrame(results)
