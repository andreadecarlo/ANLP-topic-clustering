#!/bin/bash
#SBATCH --job-name=anlp-bertopic-online
#SBATCH --partition=edu-long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=sbatches/logs/05_bertopic_online_%j.out
#SBATCH --error=sbatches/logs/05_bertopic_online_%j.err

set -e
# Use submission dir so we're in a writable project path on the compute node
PROJECT="${ANLP_PROJECT:-${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}}"
cd "$PROJECT"
echo "Working directory: $(pwd)"
mkdir -p sbatches/logs data/processed models

echo "=== ANLP: BERTopic online (full dataset) ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Load (effectively) all dataset: large max-docs, full year range from config/defaults
# Tune YEAR_MIN/YEAR_MAX and --max-docs if you want a subset (e.g. --max-docs 500000)
uv run anlp bertopic-online \
  --year-min 1960 \
  --year-max 2020 \
  --max-docs 500000 \
  --save models/bertopic_lyrics_online_full

echo "Done."
