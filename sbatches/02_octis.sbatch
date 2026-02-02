#!/bin/bash
#SBATCH --job-name=anlp-octis
#SBATCH --partition=edu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sbatches/logs/02_octis_%j.out
#SBATCH --error=sbatches/logs/02_octis_%j.err
# Optional: run after data job: --dependency=afterok:<jobid_of_01_data>

set -e
PROJECT="${ANLP_PROJECT:-$(dirname "$(dirname "$(realpath "$0")")")}"
cd "$PROJECT"
mkdir -p sbatches/logs

echo "=== ANLP: OCTIS comparison (LDA, NMF) ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"
uv run anlp octis --year-min 2010 --year-max 2020 --max-docs 50000 --num-topics 20
echo "Done."
