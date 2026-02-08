#!/bin/bash
#SBATCH --job-name=anlp-viz
#SBATCH --partition=edu-long
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=sbatches/logs/04_visualize_%j.out
#SBATCH --error=sbatches/logs/04_visualize_%j.err

set -e
PROJECT="${ANLP_PROJECT:-${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}}"
cd "$PROJECT"
echo "Working directory: $(pwd)"
mkdir -p sbatches/logs models

echo "=== ANLP: load model, document map PNG, log topics ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"
# If _docs.parquet and _reduced_embeddings.npy are missing, add e.g.:
#   --docs data/raw/song_lyrics.csv --text-column lyrics --max-docs 10000
uv run python scripts/load_visualize_save.py \
    --model models/bertopic_finetuned_30k \
    --docs data/raw/song_lyrics.csv \
    --text-column lyrics \
    --max-docs 30000 \
    --output models/bertopic_finetuned_30k_document_map.png
echo "Done."
