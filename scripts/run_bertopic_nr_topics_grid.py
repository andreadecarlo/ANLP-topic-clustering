#!/usr/bin/env python3
"""
Fit several BERTopic models with different `nr_topics` values and
append their OCTIS-style metrics to a CSV.

This automates running a small grid over `nr_topics` similar to the
OCTIS comparison, but for BERTopic only.

Examples
--------
Default grid (config, auto, 20, 50, 80 topics):

    python scripts/run_bertopic_nr_topics_grid.py

Custom grid:

    python scripts/run_bertopic_nr_topics_grid.py \
        --nr-topics None auto 20 40 60 \
        --metrics-csv models/bertopic_metrics.csv \
        --append
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Third-party
import pandas as pd

# Add project root for `anlp` imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anlp.bertopic_pipeline import fit_bertopic_on_lyrics  # type: ignore
from anlp.config import (
    BERTOPIC_NUM_TOPICS,
    MAX_DOCS_SUBSET,
    MODELS_DIR,
    YEAR_MAX,
    YEAR_MIN,
)

# Reuse metric helpers from compare_bertopic_models
from scripts.compare_bertopic_models import (  # type: ignore
    append_results_to_csv,
    compute_metrics_for_model,
)


def parse_nr_topics(value: str) -> tuple[int | str | None, str]:
    """
    Parse a user-specified nr_topics string into a Python value and a label
    used to name the model directory.

    Accepted values:
        - "config": use BERTOPIC_NUM_TOPICS from config.py
        - "None" / "none": None  (keep all clusters)
        - "auto": "auto"
        - integer string (e.g. "20", "80")
    """
    v = value.strip()
    if v.lower() == "config":
        return BERTOPIC_NUM_TOPICS, "config"
    if v.lower() == "none":
        return None, "none"
    if v.lower() == "auto":
        return "auto", "auto"
    try:
        n = int(v)
    except ValueError:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            f"Invalid nr_topics value '{value}'. Use an int, 'auto', 'None', or 'config'."
        )
    return n, f"n{n}"


def build_default_grid() -> list[str]:
    """Default grid in string form so it goes through the same parser."""
    values: list[str] = ["config", "auto", "20", "50", "80"]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit BERTopic models for multiple nr_topics values and append their "
            "OCTIS-style metrics (coherence, diversity) to a CSV."
        )
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=YEAR_MIN,
        help=f"Min year for lyrics subset (default: {YEAR_MIN})",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=YEAR_MAX,
        help=f"Max year for lyrics subset (default: {YEAR_MAX})",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=MAX_DOCS_SUBSET,
        help=f"Max documents to load (default: {MAX_DOCS_SUBSET})",
    )
    parser.add_argument(
        "--nr-topics",
        nargs="+",
        type=str,
        default=None,
        help=(
            "List of nr_topics values to try (int, 'auto', 'None', or 'config'). "
            "Default grid: config, auto, 20, 50, 80."
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=MODELS_DIR / "bertopic_metrics.csv",
        help="Output CSV for metrics (default: models/bertopic_metrics.csv)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing metrics CSV instead of overwriting.",
    )

    args = parser.parse_args()

    grid_specs = args.nr_topics or build_default_grid()

    results: list[dict] = []
    for spec in grid_specs:
        nr_topics_val, label = parse_nr_topics(spec)

        # Build model save path name
        base_name = "bertopic_lyrics"
        if label == "config":
            # Distinguish from a plain run if desired
            model_name = base_name
        else:
            model_name = f"{base_name}_{label}"
        save_path = MODELS_DIR / model_name

        print(
            f"\n=== Fitting BERTopic: nr_topics={nr_topics_val!r}, "
            f"save_path={save_path} ===",
            flush=True,
        )

        model, docs_df, topics, probs = fit_bertopic_on_lyrics(
            year_min=args.year_min,
            year_max=args.year_max,
            max_docs=args.max_docs,
            save_path=save_path,
            nr_topics=nr_topics_val,
        )
        # Avoid unused variable warnings (and signal to readers they are kept for side effects)
        _ = (model, docs_df, topics, probs)

        # Compute metrics using the saved artifacts
        try:
            metrics = compute_metrics_for_model(save_path)
        except Exception as e:  # pragma: no cover - runtime safety
            print(f"Error computing metrics for {save_path}: {e}", file=sys.stderr)
            continue

        # Also store the grid spec for clarity
        metrics["grid_spec"] = spec
        results.append(metrics)

    if not results:
        print("No metrics computed (all grid runs failed?).", file=sys.stderr)
        sys.exit(1)

    append_results_to_csv(results, args.metrics_csv, append=args.append)
    print(
        f"\nWrote metrics for {len(results)} grid setting(s) to {args.metrics_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()

