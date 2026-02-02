#!/bin/bash
#SBATCH --job-name=anlp-pipeline
#SBATCH --partition=edu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=sbatches/logs/run_all_%j.out

# Run data -> octis -> bertopic in one job (no dependencies between stages)
set -e
PROJECT="${ANLP_PROJECT:-$(dirname "$(dirname "$(realpath "$0")")")}"
cd "$PROJECT"
mkdir -p sbatches/logs data/processed models

echo "=== ANLP: full pipeline (data + OCTIS + BERTopic) ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"

echo "--- 1. Data subset ---"
uv run anlp data --year-min 2010 --year-max 2020 --max-docs 50000

echo "--- 2. OCTIS comparison ---"
uv run anlp octis --year-min 2010 --year-max 2020 --max-docs 50000 --num-topics 20

echo "--- 3. BERTopic ---"
uv run anlp bertopic --year-min 2010 --year-max 2020 --max-docs 50000 --save models/bertopic_lyrics

echo "=== Pipeline done ==="
