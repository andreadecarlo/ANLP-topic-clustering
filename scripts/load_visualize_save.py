#!/usr/bin/env python3
"""
Load a saved BERTopic model, create the document map visualization (notebook-style),
save it as PNG, and log topics with names and representative samples.

Works in two modes:
- With docs + reduced embeddings already saved: uses them (no --docs). It looks for:
  - New layout (e.g. when loading from a Hugging Face Hub snapshot):
      <model_dir>/docs.parquet
      <model_dir>/reduced_embeddings.npy
  - Legacy layout (local training artifacts):
      <parent>/<model_name>_docs.parquet
      <parent>/<model_name>_reduced_embeddings.npy
- Without those files: pass --docs <path> to a CSV/parquet with document text; script
  will run transform() and compute reduced embeddings on the fly.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root for anlp imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anlp.bertopic_pipeline import get_topic_labels, load_bertopic_model
from anlp.bertopic_viz import compute_reduced_embeddings, load_reduced_embeddings
from anlp.config import MODELS_DIR


def truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def log_topics_with_titles(
    topic_model,
    titles: list[str],
    topics: list[int],
    scores: list[float] | None = None,
    top_k: int = 10,
    focus_tid: int | None = None,
) -> None:
    """Print to stdout: topic id, name, and up to top_k most representative titles."""
    from collections import defaultdict

    labels = get_topic_labels(topic_model)
    per_topic: dict[int, list[tuple[float | None, str]]] = defaultdict(list)

    for i, tid in enumerate(topics):
        if i >= len(titles):
            continue
        title = titles[i]
        score = None
        if scores is not None and i < len(scores):
            score = scores[i]
        per_topic[tid].append((score, title))

    def _print_topic(tid: int) -> None:
        if tid == -1:
            name = labels.get(tid, "Outliers")
        else:
            name = labels.get(tid, str(tid))
        print(f"\n--- Topic {tid}: {name} ---")
        docs_for_topic = per_topic[tid]
        # Sort by score (probability) descending when available
        if scores is not None:
            docs_for_topic = sorted(
                docs_for_topic,
                key=lambda x: (-1 if x[0] is None else x[0]),
                reverse=True,
            )
        for j, (_, title) in enumerate(docs_for_topic[:top_k], 1):
            print(f"  [{j}] {title}")

    # First, optionally print the focus topic (for the requested song/title)
    if focus_tid is not None and focus_tid in per_topic:
        print("\n=== Focus topic for requested title ===")
        _print_topic(focus_tid)

    # Then, print all topics (skipping focus topic to avoid duplication)
    for tid in sorted(labels.keys()):
        if tid not in per_topic:
            continue
        if focus_tid is not None and tid == focus_tid:
            continue
        _print_topic(tid)
    print()


def build_document_map_figure(topic_model, titles: list[str], reduced_embeddings, **kwargs):
    """Same as notebook: visualize_documents with custom_labels=True and given kwargs."""
    fig = topic_model.visualize_documents(
        titles,
        reduced_embeddings=reduced_embeddings,
        hide_annotations=True,
        hide_document_hover=False,
        custom_labels=True,
        **kwargs,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load BERTopic model (local path or from Hugging Face Hub), create "
            "the document map, save as PNG, and log topics with representative samples."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "bertopic_lyrics",
        help=(
            "Path to saved BERTopic model directory when using a local model "
            "(default: models/bertopic_lyrics). Ignored when --hf-repo-id is used."
        ),
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face Hub repo id (e.g. 'username/bertopic-lyrics'). "
            "When set, the script will download the snapshot and load the model "
            "from there instead of a local path."
        ),
    )
    parser.add_argument(
        "--hf-model-dir-name",
        type=str,
        default="",
        help=(
            "Optional subfolder name inside the Hub repo that contains the BERTopic model "
            "and its artifacts. Leave empty (default) when the model files live at repo root."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PNG path (default: <model_dir>/bertopic_document_map.png)",
    )
    parser.add_argument(
        "--n-representative",
        type=int,
        default=10,
        help="Number of representative titles to log per topic (default: 10)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="PNG width in pixels (default: 1200)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="PNG height in pixels (default: 800)",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=None,
        help="Path to CSV/parquet with document text (required when _docs.parquet and _reduced_embeddings.npy are missing)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="lyrics",
        help="Column name for document text in --docs file (default: lyrics)",
    )
    parser.add_argument(
        "--title-column",
        type=str,
        default=None,
        help="Column name for titles/hover in --docs file (default: title or song, else truncate text)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Max documents to use when loading from --docs (default: all)",
    )
    parser.add_argument(
        "--focus-title",
        type=str,
        default=None,
        help="Song/title to focus on (log its topic and representative titles first)",
    )
    args = parser.parse_args()

    # Resolve model location: either from Hugging Face Hub or a local path
    if args.hf_repo_id:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print(
                "huggingface_hub is required when using --hf-repo-id. "
                "Install it with: pip install huggingface_hub (or uv add huggingface_hub).",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Downloading model snapshot from Hugging Face Hub: {args.hf_repo_id}")
        cache_dir = snapshot_download(args.hf_repo_id)
        base_path = Path(cache_dir)
        if args.hf_model_dir_name:
            model_path = base_path / args.hf_model_dir_name
        else:
            model_path = base_path
        if not model_path.exists():
            print(
                f"Downloaded repo but model directory not found: {model_path}\n"
                "If your model is inside a subfolder, pass it via --hf-model-dir-name.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Using model directory from Hub snapshot: {model_path}")
    else:
        model_path = args.model
        if not model_path.exists():
            print(f"Model path does not exist: {model_path}", file=sys.stderr)
            sys.exit(1)

    # Prefer the new, compact naming inside the model directory (works well with HF Hub snapshots),
    # but fall back to the legacy naming next to the model dir for backward compatibility.
    docs_path_new = model_path / "docs.parquet"
    reduced_path_new = model_path / "reduced_embeddings.npy"
    docs_path_legacy = model_path.parent / (model_path.name + "_docs.parquet")
    reduced_path_legacy = model_path.parent / (model_path.name + "_reduced_embeddings.npy")

    if docs_path_new.exists() and reduced_path_new.exists():
        docs_path = docs_path_new
        reduced_path = reduced_path_new
    else:
        docs_path = docs_path_legacy
        reduced_path = reduced_path_legacy

    use_saved = docs_path.exists() and reduced_path.exists()

    if not use_saved and not args.docs:
        print(
            "No saved docs + reduced embeddings found next to the model, and no --docs provided.",
            file=sys.stderr,
        )
        print(
            "Expected either:\n"
            f"  - {docs_path_new.name} and {reduced_path_new.name} inside the model directory, or\n"
            f"  - {docs_path_legacy.name} and {reduced_path_legacy.name} next to the model directory,\n"
            "or a --docs <path> to a CSV/parquet file.",
            file=sys.stderr,
        )
        sys.exit(1)
    if use_saved and args.docs:
        print("Using saved docs + reduced embeddings (ignoring --docs).")

    print("Loading model...")
    topic_model = load_bertopic_model(model_path)

    scores: list[float] | None = None

    if use_saved:
        docs_df = pd.read_parquet(docs_path)
        reduced = load_reduced_embeddings(reduced_path)
        topics = docs_df["topic"].tolist() if "topic" in docs_df.columns else getattr(topic_model, "topics_", None)
        if not topics:
            print("No topics found (parquet needs 'topic' column).", file=sys.stderr)
            sys.exit(1)
        text_col = "lyrics" if "lyrics" in docs_df.columns else docs_df.columns[0]
        documents = docs_df[text_col].tolist()
        if "topic_prob" in docs_df.columns:
            scores = docs_df["topic_prob"].astype(float).tolist()
        n = min(len(docs_df), len(topics), len(reduced))
        docs_df = docs_df.iloc[:n]
        topics = topics[:n]
        reduced = reduced[:n]
        documents = documents[:n]
        titles = docs_df.get("title", docs_df.get("song", None))
        if titles is None:
            titles = [truncate(d, 80) for d in documents]
        else:
            titles = titles.tolist()[:n]
    else:
        # Load from --docs and compute topics + reduced embeddings
        path = args.docs
        if not path.exists():
            print(f"Docs file not found: {path}", file=sys.stderr)
            sys.exit(1)
        if path.suffix.lower() == ".parquet":
            docs_df = pd.read_parquet(path)
        else:
            docs_df = pd.read_csv(path)
        if args.max_docs:
            docs_df = docs_df.iloc[: args.max_docs]
        text_col = args.text_column
        if text_col not in docs_df.columns:
            print(f"Column '{text_col}' not in {list(docs_df.columns)}", file=sys.stderr)
            sys.exit(1)
        documents = docs_df[text_col].astype(str).tolist()
        print(f"Transforming {len(documents)} documents...")
        topics, probs = topic_model.transform(documents)
        # Align model state to this document set so visualize_documents indexes correctly
        # (model may have been fit on more docs, e.g. 30k; we pass 10k -> IndexError otherwise)
        topic_model.topics_ = topics
        if probs is not None:
            topic_model.probabilities_ = probs
            # Use max probability per document as representativeness score
            try:
                scores = [float(max(row)) for row in probs]
            except TypeError:
                scores = None
        print("Computing reduced embeddings...")
        reduced = compute_reduced_embeddings(topic_model, documents)
        title_col = args.title_column or ("title" if "title" in docs_df.columns else "song" if "song" in docs_df.columns else None)
        if title_col and title_col in docs_df.columns:
            titles = docs_df[title_col].astype(str).tolist()
        else:
            titles = [truncate(d, 80) for d in documents]

    n = min(len(documents), len(topics), len(reduced), len(titles), len(scores) if scores is not None else len(documents))
    documents = documents[:n]
    topics = topics[:n]
    reduced = reduced[:n]
    titles = titles[:n]
    if scores is not None:
        scores = scores[:n]

    # Optional: determine focus topic based on a requested title
    focus_tid: int | None = None
    if args.focus_title:
        q = args.focus_title.lower()
        matches = [i for i, t in enumerate(titles) if t.lower() == q]
        if not matches:
            # fallback: substring match
            matches = [i for i, t in enumerate(titles) if q in t.lower()]
        if matches:
            idx = matches[0]
            focus_tid = topics[idx]
            print(f"Focus title: '{titles[idx]}' -> Topic {focus_tid}")
        else:
            print(f"Focus title '{args.focus_title}' not found in titles list.", file=sys.stderr)

    print("Topics with representative titles:")
    log_topics_with_titles(
        topic_model,
        titles,
        topics,
        scores=scores,
        top_k=args.n_representative,
        focus_tid=focus_tid,
    )

    print("Building document map...")
    fig = build_document_map_figure(topic_model, titles, reduced)

    out_path = args.output or (model_path.parent / (model_path.name + "_document_map.png"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.write_image(str(out_path), width=args.width, height=args.height)
        print(f"Saved: {out_path}")
    except Exception as e:
        if "kaleido" in str(e).lower() or "orca" in str(e).lower():
            html_path = out_path.with_suffix(".html")
            fig.write_html(str(html_path))
            print(f"PNG export failed (install kaleido: pip install kaleido). Wrote HTML instead: {html_path}", file=sys.stderr)
        else:
            raise


if __name__ == "__main__":
    main()
