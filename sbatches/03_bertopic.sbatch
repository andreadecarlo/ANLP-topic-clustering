#!/bin/bash
#SBATCH --job-name=anlp-bertopic
#SBATCH --partition=edu-long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
##SBATCH --time=08:00:00
#SBATCH --output=sbatches/logs/03_bertopic_%j.out

set -e
# Use submission dir so we're in a writable project path on the compute node
PROJECT="${ANLP_PROJECT:-${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}}"
cd "$PROJECT"
echo "Working directory: $(pwd)"
mkdir -p sbatches/logs data/processed models

echo "=== ANLP: BERTopic fit (human-readable topics) ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"
# Reduce CUDA fragmentation when reserved-but-unallocated memory is large
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run anlp bertopic --year-min 1960 --year-max 1970 --max-docs 20_000 --save models/bertopic_finetuned_20k  --visualize
echo "Done."
