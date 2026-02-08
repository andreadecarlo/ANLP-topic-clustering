"""BERTopic visualizations: document map, topic barchart, and static topic map (notebook-style)."""

from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic

from anlp.bertopic_pipeline import get_topic_labels


# Distinct colors for topics (notebook-style)
TOPIC_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#000000",
]


def compute_reduced_embeddings(
    topic_model: BERTopic,
    documents: list[str],
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Embed documents via the topic model's backend, reduce to 2D with UMAP.
    Returns the 2D array for visualize_documents (no file saved).
    """
    from umap import UMAP

    emb = topic_model._extract_embeddings(documents, method="document", verbose=True)
    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(emb)


def compute_and_save_reduced_embeddings(
    topic_model: BERTopic,
    documents: list[str],
    save_path: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Embed documents via the topic model's backend, reduce to 2D with UMAP, and save as .npy.
    Returns the 2D array for immediate use (tutorial: same reduced_embeddings for visualize_documents).
    Uses topic_model._extract_embeddings so any backend (SentenceTransformerBackend, etc.) works.
    """
    reduced = compute_reduced_embeddings(
        topic_model, documents,
        n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state,
    )
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, reduced)
    return reduced


def load_reduced_embeddings(path: Path | str) -> np.ndarray:
    """Load reduced embeddings from .npy file."""
    return np.load(path)


def visualize_document_map(
    topic_model: BERTopic,
    documents: list[str],
    reduced_embeddings: np.ndarray,
    out_path: Path | str,
    titles: list[str] | None = None,
    hide_annotations: bool = True,
    hide_document_hover: bool = False,
) -> None:
    """
    Interactive document map (Plotly). Save as HTML.
    Use titles for hover (e.g. song titles) instead of full lyrics to keep file small.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hover_docs = titles if titles is not None else documents
    fig = topic_model.visualize_documents(
        hover_docs,
        reduced_embeddings=reduced_embeddings,
        hide_annotations=hide_annotations,
        hide_document_hover=hide_document_hover,
        custom_labels=True,
    )
    fig.write_html(str(out_path))


def visualize_barchart(
    topic_model: BERTopic,
    out_path: Path | str,
    top_n_topics: int = 12,
    n_words: int = 5,
) -> None:
    """Topic word barchart (Plotly). Save as HTML."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = topic_model.visualize_barchart(
        top_n_topics=top_n_topics,
        n_words=n_words,
        custom_labels=True,
        title="Topic words (c-TF-IDF)",
    )
    fig.write_html(str(out_path))


def visualize_topic_map_static(
    topic_model: BERTopic,
    reduced_embeddings: np.ndarray,
    topics: list[int],
    doc_lengths: list[int],
    out_path: Path | str,
    top_n_topics: int = 50,
    x_lim: tuple[float, float] = (-10, 10),
    y_lim: tuple[float, float] = (-10, 10),
    figsize: tuple[int, int] = (12, 12),
) -> None:
    """
    Static scatter plot with topic labels at centroids (notebook-style).
    Uses seaborn scatter + adjustText for non-overlapping labels.
    """
    import itertools
    import textwrap

    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import seaborn as sns
    from adjustText import adjust_text

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = get_topic_labels(topic_model)

    colors = itertools.cycle(TOPIC_COLORS)
    color_key = {str(t): next(colors) for t in set(topics) if t != -1}

    df = pd.DataFrame(
        {
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "Topic": [str(t) for t in topics],
        }
    )
    df["Length"] = doc_lengths
    df = df.loc[df["Topic"] != "-1"]
    df = df.loc[
        (df["y"] > y_lim[0]) & (df["y"] < y_lim[1]) & (df["x"] > x_lim[0]) & (df["x"] < x_lim[1]),
        :,
    ]
    df["Topic"] = df["Topic"].astype("category")

    mean_df = df.groupby("Topic").mean().reset_index()
    mean_df["Topic"] = mean_df["Topic"].astype(int)
    mean_df = mean_df.sort_values("Topic")

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="Topic",
        palette=color_key,
        alpha=0.4,
        size="Length",
        sizes=(20, 100),
        ax=ax,
        legend=False,
    )

    texts = []
    xs, ys = [], []
    for _, row in mean_df.iterrows():
        topic = int(row["Topic"])
        if topic > top_n_topics:
            continue
        name = textwrap.fill(labels.get(topic, str(topic)), 20)
        x, y = row["x"], row["y"]
        xs.append(x)
        ys.append(y)
        t = ax.text(
            x,
            y,
            name,
            size=9,
            ha="center",
            color=color_key.get(str(topic), "gray"),
            path_effects=[pe.withStroke(linewidth=0.5, foreground="black")],
        )
        texts.append(t)

    if texts:
        adjust_text(
            texts,
            x=xs,
            y=ys,
            time_lim=1,
            force_text=(0.01, 0.02),
            force_static=(0.01, 0.02),
            force_pull=(0.5, 0.5),
            ax=ax,
        )
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_visualizations(
    topic_model: BERTopic,
    documents: list[str],
    topics: list[int],
    reduced_embeddings: np.ndarray,
    out_dir: Path | str,
    titles: list[str] | None = None,
    doc_lengths: list[int] | None = None,
) -> None:
    """
    Generate all three visualizations and save to out_dir:
    - documents.html (interactive document map)
    - barchart.html (topic words)
    - topic_map.png (static scatter with labels)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if doc_lengths is None:
        doc_lengths = [len(d) for d in documents]
    if titles is None and hasattr(topic_model, "custom_labels_"):
        titles = documents  # fallback to first N chars in HTML if needed

    visualize_document_map(
        topic_model,
        documents,
        reduced_embeddings,
        out_dir / "documents.html",
        titles=titles,
    )
    visualize_barchart(topic_model, out_dir / "barchart.html")
    visualize_topic_map_static(
        topic_model,
        reduced_embeddings,
        topics,
        doc_lengths,
        out_dir / "topic_map.png",
    )
