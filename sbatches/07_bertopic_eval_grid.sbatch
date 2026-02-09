#!/bin/bash
#SBATCH --job-name=anlp-bertopic-grid
#SBATCH --partition=edu-thesis
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
##SBATCH --time=24:00:00
#SBATCH --output=sbatches/logs/07_bertopic_eval_grid_%j.out
#SBATCH --error=sbatches/logs/07_bertopic_eval_grid_%j.err

set -e
# Use submission dir so we're in a writable project path on the compute node
PROJECT="${ANLP_PROJECT:-${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}}"
cd "$PROJECT"
echo "Working directory: $(pwd)"
mkdir -p sbatches/logs data/processed models

echo "=== ANLP: BERTopic nr_topics grid (fit + metrics) ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Default grid: config, auto, 20, 50, 80 topics on the same lyrics subset.
# Adjust --year-min/--year-max/--max-docs or the grid as needed.
uv run python scripts/run_bertopic_nr_topics_grid.py \
  --year-min 1960 \
  --year-max 1980 \
  --max-docs 80_000 \
  --nr-topics auto None 20 100 200 400\
  --metrics-csv models/bertopic_metrics.csv \
  --append

echo "Done."

