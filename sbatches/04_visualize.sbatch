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
    --hf-repo-id Dr3dre/bertopic-lyrics-auto \
    --docs data/raw/song_lyrics.csv \
    --text-column lyrics \
    --max-docs 200_000 \
    --output models/bertopic_finetuned_200k_alltopics_document_map.png \
    --n-representative 10 \
    --title-column title \
    --focus-title "We Are the World"

echo "Done."
# --model models/bertopic_finetuned_200k_alltopics \
