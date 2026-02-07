"""CLI for topic modeling: OCTIS comparison, BERTopic, and retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anlp.config import (
    MAX_DOCS_SUBSET,
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
    p_bert.set_defaults(func=cmd_bertopic)

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

    df = load_lyrics_subset(
        year_min=args.year_min,
        year_max=args.year_max,
        max_docs=args.max_docs,
        csv_path=args.csv,
    )
    print(f"Loaded {len(df)} documents (years {args.year_min}-{args.year_max})")
    print(df.head())


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

    model, docs_df, topics, probs = fit_bertopic_on_lyrics(
        year_min=args.year_min,
        year_max=args.year_max,
        max_docs=args.max_docs,
        save_path=args.save,
    )
    labels = get_topic_labels(model)
    print(f"Fitted BERTopic: {len(set(topics)) - (1 if -1 in topics else 0)} topics")
    print("Topic labels (sample):", list(labels.items())[:5])


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
