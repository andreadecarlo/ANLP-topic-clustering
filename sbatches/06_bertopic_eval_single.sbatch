#!/bin/bash
#SBATCH --job-name=anlp-bertopic-eval
#SBATCH --partition=edu-long
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=sbatches/logs/06_bertopic_eval_single_%j.out
#SBATCH --error=sbatches/logs/06_bertopic_eval_single_%j.err

set -e
# Use submission dir so we're in a writable project path on the compute node
PROJECT="${ANLP_PROJECT:-${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}}"
cd "$PROJECT"
echo "Working directory: $(pwd)"
mkdir -p sbatches/logs models

echo "=== ANLP: BERTopic single-model evaluation (OCTIS metrics) ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"

# Change --model to point to the BERTopic model you want to evaluate.
# By default, this evaluates the main bertopic_lyrics model and writes/updates
# models/bertopic_metrics.csv.
uv run python scripts/compare_bertopic_models.py \
  --model models/bertopic_lyrics \
  --csv models/bertopic_metrics.csv \
  --append

echo "Done."

