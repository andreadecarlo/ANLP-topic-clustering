"""CLI for topic modeling: OCTIS comparison, BERTopic, and retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anlp.config import (
    BERTOPIC_NUM_TOPICS,
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
    p_bert.add_argument(
        "--nr-topics",
        type=str,
        default=str(BERTOPIC_NUM_TOPICS)
        if BERTOPIC_NUM_TOPICS is not None
        else "None",
        help=(
            "BERTopic nr_topics parameter (int, 'auto', or 'None' to keep all clusters). "
            f"Default: {BERTOPIC_NUM_TOPICS!r}."
        ),
    )
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
    p_viz = subparsers.add_parser(
        "visualize",
        help="Generate BERTopic visualizations from saved model (local path or Hugging Face repo ID)",
    )
    p_viz.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to BERTopic model dir or Hugging Face repo ID (e.g. 'Dr3dre/bertopic-lyrics-auto')",
    )
    p_viz.add_argument("--out-dir", type=Path, default=None, help="Output directory for HTML/PNG (default: next to model)")
    p_viz.set_defaults(func=cmd_visualize)

    # Similar songs
    p_similar = subparsers.add_parser(
        "similar",
        help="Get similar songs for a song (by title or doc index)",
    )
    p_similar.add_argument(
        "query",
        help="Song title (substring match, case-insensitive) or document index (row in processed corpus)",
    )
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

    # Parse nr_topics: allow int, 'auto', or 'None'
    nr_topics_arg: str = getattr(args, "nr_topics", "None")
    if nr_topics_arg.lower() == "none":
        nr_topics_val: int | str | None = None
    elif nr_topics_arg.lower() == "auto":
        nr_topics_val = "auto"
    else:
        nr_topics_val = int(nr_topics_arg)

    model, docs_df, topics, probs = fit_bertopic_on_lyrics(
        year_min=args.year_min,
        year_max=args.year_max,
        max_docs=args.max_docs,
        save_path=save_path,
        nr_topics=nr_topics_val,
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

    # Allow either numeric doc index or title substring as query
    query = str(args.query)
    doc_id: int | None = None
    if query.isdigit():
        doc_id = int(query)
    else:
        if "title" not in docs_df.columns:
            print(
                "Docs parquet has no 'title' column; similar-by-title is unavailable. "
                "Use a numeric document index instead.",
                file=sys.stderr,
            )
            sys.exit(1)
        # Case-insensitive substring match on title
        mask = docs_df["title"].astype(str).str.contains(query, case=False, na=False)
        matches = docs_df[mask]
        if matches.empty:
            print(f"No songs found with title containing {query!r}.", file=sys.stderr)
            sys.exit(1)
        # Use the first match; print a hint if there are multiple
        if len(matches) > 1:
            print(
                f"Multiple songs match {query!r}; using the first match: "
                f"{matches.iloc[0].get('title', '')!r} by {matches.iloc[0].get('artist', '')!r}.",
                file=sys.stderr,
            )
        doc_id = int(matches.index[0])

    topics = model.topics_
    probs = getattr(model, "probabilities_", None)
    out = similar_songs_for_song(doc_id, docs_df, topics, probs, top_k=args.top_k)
    print(out.to_string(index=False))


def cmd_visualize(args: argparse.Namespace) -> None:
    from anlp.bertopic_pipeline import load_bertopic_model
    from anlp.bertopic_viz import load_reduced_embeddings, run_visualizations

    model_arg = str(args.model)
    
    # Check if model_arg looks like a Hugging Face repo ID (contains / and doesn't exist as path)
    # If so, download it first
    model_path = Path(model_arg)
    if "/" in model_arg and not model_path.exists() and not model_path.is_absolute():
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print(
                "huggingface_hub is required when using Hugging Face repo IDs. "
                "Install it with: pip install huggingface_hub (or uv add huggingface_hub).",
                file=sys.stderr,
            )
            sys.exit(1)
        
        repo_id = model_arg
        print(f"Downloading model from Hugging Face Hub: {repo_id}")
        cache_dir = snapshot_download(repo_id=repo_id)
        model_path = Path(cache_dir)
        print(f"Model downloaded to: {model_path}")
    
    if not model_path.exists():
        print(f"Model path does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    model = load_bertopic_model(model_path)
    
    # Try new layout first (docs.parquet and reduced_embeddings.npy inside model dir)
    docs_path_new = model_path / "docs.parquet"
    reduced_path_new = model_path / "reduced_embeddings.npy"
    # Fall back to legacy layout (next to model dir)
    docs_path_legacy = model_path.parent / (model_path.name + "_docs.parquet")
    reduced_path_legacy = model_path.parent / (model_path.name + "_reduced_embeddings.npy")
    
    if docs_path_new.exists() and reduced_path_new.exists():
        docs_path = docs_path_new
        reduced_path = reduced_path_new
    elif docs_path_legacy.exists() and reduced_path_legacy.exists():
        docs_path = docs_path_legacy
        reduced_path = reduced_path_legacy
    else:
        print(f"Docs parquet not found at {docs_path_new} or {docs_path_legacy}", file=sys.stderr)
        print(f"Reduced embeddings not found at {reduced_path_new} or {reduced_path_legacy}", file=sys.stderr)
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
