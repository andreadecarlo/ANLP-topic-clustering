#!/bin/bash
#SBATCH --job-name=anlp-data
#SBATCH --partition=edu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=sbatches/logs/01_data_%j.out
#SBATCH --error=sbatches/logs/01_data_%j.err

set -e
# Use submission dir so we're in a writable project path on the compute node
PROJECT="${ANLP_PROJECT:-${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}}"
cd "$PROJECT"
echo "Working directory: $(pwd)"
mkdir -p sbatches/logs data/processed models

echo "=== ANLP: prepare data subset ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID  Partition: $SLURM_JOB_PARTITION"
uv run anlp data --year-min 2010 --year-max 2020 --max-docs 50000
echo "Done."
