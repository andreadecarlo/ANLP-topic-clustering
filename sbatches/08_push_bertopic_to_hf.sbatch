#!/bin/bash
#SBATCH --job-name=anlp-push-hf
#SBATCH --partition=edu-long
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=sbatches/logs/08_push_bertopic_to_hf_%j.out
#SBATCH --error=sbatches/logs/08_push_bertopic_to_hf_%j.err

set -e

# Use submission dir so we're in a writable project path on the compute node
PROJECT="${ANLP_PROJECT:-${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}}"
cd "$PROJECT"
echo "Working directory: $(pwd)"
mkdir -p sbatches/logs models

echo "=== ANLP: Push BERTopic model + artifacts to Hugging Face Hub ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"

HF_REPO_ID="${HF_REPO_ID:-Dr3dre/bertopic-lyrics-n200}"

echo "Pushing to Hugging Face repo: $HF_REPO_ID"

# This script expects that:
#   - models/bertopic_lyrics/           (or another model dir) exists
#   - models/bertopic_lyrics_docs.parquet
#   - models/bertopic_lyrics_reduced_embeddings.npy
# have been created by the BERTopic pipeline.
# Adjust --model-dir/--docs-parquet/--reduced-embeddings below if needed.

uv run python scripts/push_bertopic_to_hf.py \
  --repo-id "$HF_REPO_ID" \
  --model-dir models/bertopic_lyrics_n200

echo "Done."

