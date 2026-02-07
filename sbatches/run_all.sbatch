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
# Use submission dir so we're in a writable project path on the compute node
PROJECT="${ANLP_PROJECT:-${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}}"
cd "$PROJECT"
echo "Working directory: $(pwd)"
mkdir -p sbatches/logs data/processed models

echo "=== ANLP: full pipeline (data + OCTIS + BERTopic) ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"

echo "--- 1. Data subset ---"
uv run anlp data 
# --year-min 2010 --year-max 2020 --max-docs 1000

echo "--- 2. BERTopic (fit and save) ---"
uv run anlp bertopic   --save models/bertopic_lyrics
# --year-min 2010 --year-max 2020 --max-docs 1000

echo "--- 3. OCTIS comparison (LDA, NMF, CTM, and saved BERTopic with same metrics) ---"
uv run anlp octis   --bertopic --bertopic-model models/bertopic_lyrics
# --year-min 2010 --year-max 2020 --max-docs 1000 --num-topics 5

echo "=== Pipeline done ==="
