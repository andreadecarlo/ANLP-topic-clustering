#!/usr/bin/env python3
"""
Compute OCTIS-style metrics (coherence, topic diversity) for one or more
saved BERTopic models and append the results to a CSV.

Usage examples
--------------
Evaluate the default BERTopic model and write a new CSV:

    python scripts/compare_bertopic_models.py \
        --model models/bertopic_lyrics \
        --csv models/bertopic_metrics.csv

Append metrics for another model (e.g. trained with a different nr_topics)
to the same CSV:

    python scripts/compare_bertopic_models.py \
        --model models/bertopic_lyrics_n50 \
        --csv models/bertopic_metrics.csv \
        --append

By default the script looks for `<model_dir>_docs.parquet` next to the model,
which is what `fit_bertopic_on_lyrics*` already saves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root for `anlp` imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anlp.bertopic_pipeline import load_bertopic_model
from anlp.config import MODELS_DIR
from anlp.data.load_lyrics import tokenize_for_octis
from anlp.octis_compare import (
    bertopic_to_octis_output,
    evaluate_octis,
)


def compute_metrics_for_model(
    model_dir: Path,
    text_column: str = "lyrics",
    topk_words: int = 10,
) -> dict:
    """
    Load a BERTopic model + its docs parquet and compute OCTIS-style metrics.

    Returns a dict with:
        - model: model_dir.name
        - nr_topics_param: BERTopic's `nr_topics` parameter (if available)
        - n_topics_found: number of non-outlier topics in the model
        - coherence_npmi: OCTIS c_npmi coherence
        - topic_diversity: OCTIS topic diversity
    """
    model_dir = model_dir.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Support both legacy naming and the new, compact layout used on the HF Hub.
    docs_path_new = model_dir / "docs.parquet"
    docs_path_legacy = model_dir.parent / (model_dir.name + "_docs.parquet")
    if docs_path_new.exists():
        docs_path = docs_path_new
    else:
        docs_path = docs_path_legacy

    if not docs_path.exists():
        raise FileNotFoundError(
            "Docs parquet not found for model.\n"
            f"  Tried: {docs_path_new}\n"
            f"  and:   {docs_path_legacy}\n"
            "Expected file produced by fit_bertopic_on_lyrics* or included in the HF snapshot."
        )

    topic_model = load_bertopic_model(model_dir)
    docs_df = pd.read_parquet(docs_path)

    if text_column not in docs_df.columns:
        # Fallback: first string-like column
        for col in docs_df.columns:
            if pd.api.types.is_string_dtype(docs_df[col].dtype):
                text_column = col
                break
        else:
            raise ValueError(
                f"Text column '{text_column}' not found in {list(docs_df.columns)} "
                "and no suitable string column was detected."
            )

    corpus = docs_df[text_column].astype(str).tolist()
    tokenized_corpus = tokenize_for_octis(corpus)

    # Convert BERTopic topics to OCTIS format and compute metrics
    output = bertopic_to_octis_output(topic_model, topk=topk_words)
    metrics = evaluate_octis(output, tokenized_corpus=tokenized_corpus, topk=topk_words)

    try:
        params = topic_model.get_params(deep=False)
        nr_topics_param = params.get("nr_topics", None)
    except Exception:
        nr_topics_param = None

    n_topics_found = len(output.get("topics", []))

    return {
        "model": model_dir.name,
        "nr_topics_param": nr_topics_param,
        "n_topics_found": n_topics_found,
        **metrics,
    }


def append_results_to_csv(results: list[dict], csv_path: Path, append: bool) -> None:
    df = pd.DataFrame(results)
    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if append and csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute OCTIS-style metrics (coherence, topic diversity) for one or more "
            "saved BERTopic models (local or from Hugging Face Hub) and append them to a CSV."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        nargs="+",
        required=False,
        help=(
            "Path(s) to BERTopic model directories when evaluating local models. "
            "If omitted and --hf-repo-id is not set, defaults to models/bertopic_lyrics."
        ),
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face Hub repo id (e.g. 'username/bertopic-lyrics'). "
            "When set, models will be loaded from a downloaded snapshot instead of local paths."
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
        "--csv",
        type=Path,
        default=MODELS_DIR / "bertopic_metrics.csv",
        help="Output CSV path (default: models/bertopic_metrics.csv)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV (default overwrites).",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="lyrics",
        help="Text column in *_docs.parquet to tokenize (default: lyrics).",
    )
    parser.add_argument(
        "--topk-words",
        type=int,
        default=10,
        help="Number of top words per topic for metrics (default: 10).",
    )

    args = parser.parse_args()

    # Resolve model directories: either from Hugging Face Hub or local paths.
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
            model_dir = base_path / args.hf_model_dir_name
        else:
            model_dir = base_path
        if not model_dir.exists():
            print(
                f"Downloaded repo but model directory not found: {model_dir}\n"
                "If your model is inside a subfolder, pass it via --hf-model-dir-name.",
                file=sys.stderr,
            )
            sys.exit(1)
        # When evaluating from Hub we typically just have one model directory
        model_dirs = [model_dir]
        print(f"Using model directory from Hub snapshot: {model_dir}")
    else:
        model_dirs = args.model or [MODELS_DIR / "bertopic_lyrics"]

    results: list[dict] = []
    for m in model_dirs:
        try:
            metrics = compute_metrics_for_model(
                m,
                text_column=args.text_column,
                topk_words=args.topk_words,
            )
        except Exception as e:
            print(f"Error processing model {m}: {e}", file=sys.stderr)
            continue
        results.append(metrics)

    if not results:
        print("No metrics computed (all models failed?).", file=sys.stderr)
        sys.exit(1)

    append_results_to_csv(results, args.csv, append=args.append)
    print(f"Wrote metrics for {len(results)} model(s) to {args.csv}")


if __name__ == "__main__":
    main()

