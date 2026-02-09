#!/usr/bin/env python3
"""
Push a local BERTopic model (plus docs parquet and reduced embeddings) to the Hugging Face Hub.

Usage (from project root):

  # 1) Ensure you have a valid HF token configured (once):
  #    - Run: huggingface-cli login
  #      or set HF_TOKEN in your environment.
  #
  # 2) Install the dependency (if not already available):
  #    uv add huggingface_hub
  #
  # 3) Push your main model (defaults to models/bertopic_lyrics + sidecar files):
  #
  #    uv run python scripts/push_bertopic_to_hf.py \\
  #        --repo-id <username>/bertopic-lyrics
  #
  #    You can override the model/docs/embeddings paths via CLI flags if needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_file, upload_folder

# Allow running from anywhere inside the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anlp.config import MODELS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Push a local BERTopic model (directory) plus its _docs.parquet and "
            "_reduced_embeddings.npy sidecar files to the Hugging Face Hub."
        )
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face repo id, e.g. 'username/bertopic-lyrics'.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODELS_DIR / "bertopic_lyrics",
        help="Path to the saved BERTopic model directory (default: models/bertopic_lyrics).",
    )
    parser.add_argument(
        "--docs-parquet",
        type=Path,
        default=None,
        help=(
            "Path to the documents parquet file. "
            "Defaults to <model_dir>_docs.parquet next to the model directory. "
            "On the Hub it will be stored at the repo root as docs.parquet."
        ),
    )
    parser.add_argument(
        "--reduced-embeddings",
        type=Path,
        default=None,
        help=(
            "Path to the reduced embeddings .npy file. "
            "Defaults to <model_dir>_reduced_embeddings.npy next to the model directory. "
            "On the Hub it will be stored at the repo root as reduced_embeddings.npy."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repo as private (default: public).",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Target branch on the Hub repo (default: main).",
    )
    parser.add_argument(
        "--commit-message",
        default="Add BERTopic model and artifacts",
        help="Commit message to use on the Hub (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir: Path = args.model_dir
    if not model_dir.exists() or not model_dir.is_dir():
        print(f"Model directory does not exist or is not a directory: {model_dir}", file=sys.stderr)
        sys.exit(1)

    # Infer sidecar paths if not explicitly provided
    docs_parquet: Path = args.docs_parquet or (model_dir.parent / f"{model_dir.name}_docs.parquet")
    reduced_embeddings: Path = args.reduced_embeddings or (
        model_dir.parent / f"{model_dir.name}_reduced_embeddings.npy"
    )

    missing: list[str] = []
    for p, label in [
        (docs_parquet, "docs parquet"),
        (reduced_embeddings, "reduced embeddings"),
    ]:
        if not p.exists():
            missing.append(f"{label}: {p}")

    if missing:
        print(
            "Some expected sidecar files are missing:\n  " + "\n  ".join(missing),
            file=sys.stderr,
        )
        print(
            "Make sure you've run the BERTopic pipeline so that _docs.parquet and "
            "_reduced_embeddings.npy are saved next to the model directory, or pass "
            "explicit paths via --docs-parquet/--reduced-embeddings.",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_id: str = args.repo_id
    print(f"Creating (or reusing) Hub repo: {repo_id} (private={args.private})")
    api = HfApi()
    # exist_ok=True so we can re-run to update the same repo
    api.create_repo(repo_id=repo_id, private=bool(args.private), exist_ok=True, repo_type="model")

    # Upload the BERTopic model directory contents directly into the repo root
    print(f"Uploading model directory contents: {model_dir} -> {repo_id}/")
    upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
        path_in_repo=".",  # no extra folder; put files at repo root
        repo_type="model",
        commit_message=args.commit_message,
    )

    # Upload sidecar artifacts at the repo root with simplified names
    for local_path, short_name in [
        (docs_parquet, "docs.parquet"),
        (reduced_embeddings, "reduced_embeddings.npy"),
    ]:
        path_in_repo = short_name
        print(f"Uploading artifact: {local_path} -> {repo_id}/{path_in_repo}")
        upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_type="model",
            commit_message=args.commit_message,
        )

    print()
    print("Done. You can now load from the Hub, for example:")
    print()
    print("  from bertopic import BERTopic")
    print("  from huggingface_hub import snapshot_download")
    print()
    print(f"  repo_id = '{repo_id}'")
    print("  local_dir = snapshot_download(repo_id)")
    print("  # Model directory is the repo root:")
    print("  model = BERTopic.load(local_dir)")
    print("  # Docs parquet: local_dir + '/docs.parquet'")
    print("  # Reduced embeddings: local_dir + '/reduced_embeddings.npy'")


if __name__ == '__main__':
    main()

