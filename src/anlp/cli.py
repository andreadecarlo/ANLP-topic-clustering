"""CLI for topic modeling: OCTIS comparison, BERTopic, and retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anlp.config import (
    MAX_DOCS_SUBSET,
    MODELS_DIR,
    OCTIS_NUM_TOPICS,
    TOP_K_REPRESENTATIVE_SONGS,
    TOP_K_SIMILAR_SONGS,
    YEAR_MAX,
    YEAR_MIN,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Topic modeling for song lyrics: OCTIS comparison, BERTopic, similar/representative songs.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Download / prepare data
    p_data = subparsers.add_parser("data", help="Download or prepare lyrics subset")
    p_data.add_argument("--year-min", type=int, default=YEAR_MIN, help="Min year")
    p_data.add_argument("--year-max", type=int, default=YEAR_MAX, help="Max year")
    p_data.add_argument("--max-docs", type=int, default=MAX_DOCS_SUBSET, help="Max documents in subset")
    p_data.add_argument("--csv", type=Path, default=None, help="Path to CSV (optional)")
    p_data.set_defaults(func=cmd_data)

    # OCTIS compare
    p_octis = subparsers.add_parser("octis", help="Compare OCTIS topic models (LDA, NMF, LSI) and optionally BERTopic")
    p_octis.add_argument("--year-min", type=int, default=YEAR_MIN)
    p_octis.add_argument("--year-max", type=int, default=YEAR_MAX)
    p_octis.add_argument("--max-docs", type=int, default=MAX_DOCS_SUBSET)
    p_octis.add_argument("--num-topics", type=int, default=OCTIS_NUM_TOPICS)
    p_octis.add_argument("--algorithms", nargs="+", default=None, help="OCTIS algorithms (default: LDA, NMF, CTM from config)")
    p_octis.add_argument("--bertopic", action="store_true", help="Also evaluate BERTopic with OCTIS metrics")
    p_octis.add_argument("--bertopic-model", type=Path, default=None, help="Path to saved BERTopic model (use instead of fitting)")
    p_octis.set_defaults(func=cmd_octis)

    # BERTopic fit
    p_bert = subparsers.add_parser("bertopic", help="Fit BERTopic and save model")
    p_bert.add_argument("--year-min", type=int, default=YEAR_MIN)
    p_bert.add_argument("--year-max", type=int, default=YEAR_MAX)
    p_bert.add_argument("--max-docs", type=int, default=MAX_DOCS_SUBSET)
    p_bert.add_argument("--save", type=Path, default=None, help="Model save path")
    p_bert.add_argument("--visualize", action="store_true", help="Generate visualizations after fitting")
    p_bert.set_defaults(func=cmd_bertopic)

    # BERTopic online (incremental) fit
    p_bert_online = subparsers.add_parser(
        "bertopic-online",
        help="Fit BERTopic incrementally (partial_fit over chunks); good for large corpora",
    )
    p_bert_online.add_argument("--year-min", type=int, default=YEAR_MIN)
    p_bert_online.add_argument("--year-max", type=int, default=YEAR_MAX)
    p_bert_online.add_argument("--max-docs", type=int, default=MAX_DOCS_SUBSET)
    p_bert_online.add_argument("--save", type=Path, default=None, help="Model save path (default: models/bertopic_lyrics_online)")
    p_bert_online.add_argument(
        "--refine-representations",
        action="store_true",
        help="Apply update_topics and Llama labels after partial_fit (default). Use this to explicitly request Llama representation.",
    )
    p_bert_online.add_argument("--no-refine", action="store_true", help="Skip update_topics and Llama labels after partial_fit (c-TF-IDF only)")
    p_bert_online.set_defaults(func=cmd_bertopic_online)

    # BERTopic visualizations (from saved model + docs + reduced embeddings)
    p_viz = subparsers.add_parser("visualize", help="Generate BERTopic visualizations from saved model")
    p_viz.add_argument("--model", type=Path, required=True, help="Path to BERTopic model dir")
    p_viz.add_argument("--out-dir", type=Path, default=None, help="Output directory for HTML/PNG (default: next to model)")
    p_viz.set_defaults(func=cmd_visualize)

    # Similar songs
    p_similar = subparsers.add_parser("similar", help="Get similar songs for a song (by doc index)")
    p_similar.add_argument("doc_id", type=int, help="Document index (row in processed corpus)")
    p_similar.add_argument("--model", type=Path, required=True, help="Path to BERTopic model dir")
    p_similar.add_argument("--top-k", type=int, default=TOP_K_SIMILAR_SONGS)
    p_similar.set_defaults(func=cmd_similar)

    # Representative songs for topic
    p_repr = subparsers.add_parser("representative", help="Get most representative songs for a topic")
    p_repr.add_argument("topic_id", type=int, help="Topic ID")
    p_repr.add_argument("--model", type=Path, required=True, help="Path to BERTopic model dir")
    p_repr.add_argument("--top-k", type=int, default=TOP_K_REPRESENTATIVE_SONGS)
    p_repr.set_defaults(func=cmd_representative)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


def cmd_data(args: argparse.Namespace) -> None:
    from anlp.data.load_lyrics import load_lyrics_subset

    dataset = load_lyrics_subset(
        year_min=args.year_min,
        year_max=args.year_max,
        max_docs=args.max_docs,
        csv_path=args.csv,
    )
    print(f"Loaded {len(dataset)} documents (years {args.year_min}-{args.year_max})")
    n_show = min(5, len(dataset))
    if n_show:
        print(dataset.select(range(n_show)).to_pandas().head())


def cmd_octis(args: argparse.Namespace) -> None:
    from anlp.octis_compare import compare_octis

    results = compare_octis(
        year_min=args.year_min,
        year_max=args.year_max,
        max_docs=args.max_docs,
        algorithms=args.algorithms,
        num_topics=args.num_topics,
        include_bertopic=args.bertopic,
        bertopic_model_path=args.bertopic_model,
    )
    print(results.to_string(index=False))


def cmd_bertopic(args: argparse.Namespace) -> None:
    from anlp.bertopic_pipeline import fit_bertopic_on_lyrics
    from anlp.bertopic_pipeline import get_topic_labels
    from anlp.bertopic_viz import load_reduced_embeddings, run_visualizations

    save_path = args.save or (MODELS_DIR / "bertopic_lyrics")
    save_path = Path(save_path)

    model, docs_df, topics, probs = fit_bertopic_on_lyrics(
        year_min=args.year_min,
        year_max=args.year_max,
        max_docs=args.max_docs,
        save_path=save_path,
    )
    labels = get_topic_labels(model)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"Fitted BERTopic: {n_topics} topics")
    print("Topic labels:")
    for tid in sorted(labels.keys()):
        print(f"  {tid}: {labels[tid]}")

    if args.visualize:
        reduced_path = save_path.parent / (save_path.name + "_reduced_embeddings.npy")
        if reduced_path.exists():
            reduced = load_reduced_embeddings(reduced_path)
            out_dir = save_path.parent / (save_path.name + "_viz")
            titles = docs_df.get("title", docs_df.get("song", None))
            if titles is None and "lyrics" in docs_df.columns:
                titles = docs_df["lyrics"].str.slice(0, 80).tolist()
            else:
                titles = titles.tolist() if hasattr(titles, "tolist") else list(titles)
            run_visualizations(
                model,
                docs_df["lyrics"].tolist(),
                topics,
                reduced,
                out_dir,
                titles=titles,
                doc_lengths=docs_df["lyrics"].str.len().tolist(),
            )
            print(f"Visualizations saved to {out_dir}")
        else:
            print("Reduced embeddings not found; skipping visualizations.", file=sys.stderr)


def cmd_bertopic_online(args: argparse.Namespace) -> None:
    from anlp.bertopic_pipeline import fit_bertopic_on_lyrics_online, get_topic_labels

    save_path = args.save or (MODELS_DIR / "bertopic_lyrics_online")
    save_path = Path(save_path)

    # Llama representation: default True unless --no-refine
    refine = getattr(args, "refine_representations", False) or not args.no_refine
    model, docs_df, topics, probs = fit_bertopic_on_lyrics_online(
        year_min=args.year_min,
        year_max=args.year_max,
        max_docs=args.max_docs,
        save_path=save_path,
        refine_representations=refine,
    )
    labels = get_topic_labels(model)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"Fitted BERTopic (online): {n_topics} topics, saved to {save_path}")
    print("Topic labels:")
    for tid in sorted(labels.keys()):
        print(f"  {tid}: {labels[tid]}")


def cmd_similar(args: argparse.Namespace) -> None:
    from anlp.bertopic_pipeline import load_bertopic_model
    from anlp.retrieval import similar_songs_for_song
    import pandas as pd

    model_path = Path(args.model)
    model = load_bertopic_model(model_path)
    docs_path = model_path.parent / (model_path.name + "_docs.parquet")
    if not docs_path.exists():
        print("Docs parquet not found at", docs_path, file=sys.stderr)
        sys.exit(1)
    docs_df = pd.read_parquet(docs_path)
    topics = model.topics_
    probs = getattr(model, "probabilities_", None)
    out = similar_songs_for_song(
        args.doc_id, docs_df, topics, probs, top_k=args.top_k
    )
    print(out.to_string(index=False))


def cmd_visualize(args: argparse.Namespace) -> None:
    from anlp.bertopic_pipeline import load_bertopic_model
    from anlp.bertopic_viz import load_reduced_embeddings, run_visualizations

    model_path = Path(args.model)
    model = load_bertopic_model(model_path)
    docs_path = model_path.parent / (model_path.name + "_docs.parquet")
    reduced_path = model_path.parent / (model_path.name + "_reduced_embeddings.npy")
    if not docs_path.exists():
        print("Docs parquet not found at", docs_path, file=sys.stderr)
        sys.exit(1)
    if not reduced_path.exists():
        print("Reduced embeddings not found at", reduced_path, file=sys.stderr)
        sys.exit(1)

    import pandas as pd

    docs_df = pd.read_parquet(docs_path)
    reduced = load_reduced_embeddings(reduced_path)
    # Use topics saved with docs (parquet has "topic" column from fit)
    if "topic" in docs_df.columns:
        topics = docs_df["topic"].tolist()
    else:
        topics = getattr(model, "topics_", None)
    if not topics:
        print("No topics found (parquet needs 'topic' column or model.topics_).", file=sys.stderr)
        sys.exit(1)
    # Align: same length
    n = min(len(docs_df), len(topics), len(reduced))
    docs_df = docs_df.iloc[:n]
    topics = topics[:n]
    reduced = reduced[:n]

    out_dir = args.out_dir or (model_path.parent / (model_path.name + "_viz"))
    out_dir = Path(out_dir)
    titles = docs_df.get("title", docs_df.get("song", None))
    if titles is None:
        titles = docs_df["lyrics"].str.slice(0, 80).tolist() if "lyrics" in docs_df.columns else None
    else:
        titles = titles.tolist()
    run_visualizations(
        model,
        docs_df["lyrics"].tolist(),
        topics,
        reduced,
        out_dir,
        titles=titles,
        doc_lengths=docs_df["lyrics"].str.len().tolist(),
    )
    print(f"Visualizations saved to {out_dir}")


def cmd_representative(args: argparse.Namespace) -> None:
    from anlp.bertopic_pipeline import load_bertopic_model
    from anlp.retrieval import representative_songs_for_topic
    import pandas as pd

    model_path = Path(args.model)
    model = load_bertopic_model(model_path)
    docs_path = model_path.parent / (model_path.name + "_docs.parquet")
    if not docs_path.exists():
        print("Docs parquet not found at", docs_path, file=sys.stderr)
        sys.exit(1)
    docs_df = pd.read_parquet(docs_path)
    topics = model.topics_
    probs = getattr(model, "probabilities_", None)
    out = representative_songs_for_topic(
        args.topic_id, docs_df, topics, probs, top_k=args.top_k
    )
    print(out.to_string(index=False))
