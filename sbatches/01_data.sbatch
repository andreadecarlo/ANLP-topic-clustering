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
PROJECT="${ANLP_PROJECT:-$(dirname "$(dirname "$(realpath "$0")")")}"
cd "$PROJECT"
mkdir -p sbatches/logs

echo "=== ANLP: prepare data subset ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID  Partition: $SLURM_JOB_PARTITION"
uv run anlp data --year-min 2010 --year-max 2020 --max-docs 50000
echo "Done."
