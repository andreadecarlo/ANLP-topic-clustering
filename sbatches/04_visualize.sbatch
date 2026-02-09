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

echo "=== ANLP: Load model from Hugging Face, create document map PNG, log topics ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"

# Load pre-trained model from Hugging Face Hub
# The model repository includes docs.parquet and reduced_embeddings.npy
# so no need to pass --docs unless you want to use different documents
HF_REPO_ID="Dr3dre/bertopic-lyrics-auto"

uv run python scripts/load_visualize_save.py \
    --hf-repo-id "${HF_REPO_ID}" \
    --output models/bertopic_hf_document_map.png \
    --n-representative 10 \
    --focus-title "We Are the World"

echo "Done."
