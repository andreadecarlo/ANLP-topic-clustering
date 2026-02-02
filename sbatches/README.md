# SLURM jobs (sbatches)

Submit from the **project root** so paths resolve correctly.

## One-shot pipeline

```bash
cd /path/to/ANLP
sbatch sbatches/run_all.sbatch
```

Runs: data subset → OCTIS comparison → BERTopic fit in a single job (~14h limit).

## Step-by-step (separate jobs)

1. **Prepare data** (creates cached subset in `data/processed/`):
   ```bash
   sbatch sbatches/01_data.sbatch
   ```

2. **OCTIS comparison** (after data exists; optional dependency):
   ```bash
   sbatch sbatches/02_octis.sbatch
   # or: sbatch --dependency=afterok:<JOBID> sbatches/02_octis.sbatch
   ```

3. **BERTopic fit** (after data exists):
   ```bash
   sbatch sbatches/03_bertopic.sbatch
   # or: sbatch --dependency=afterok:<JOBID> sbatches/03_bertopic.sbatch
   ```

## Customisation

- **Partition / resources**: Edit the `#SBATCH` lines in each file (e.g. `--partition`, `--mem`, `--time`, `--cpus-per-task`).
- **Project path**: Set `ANLP_PROJECT` if you submit from another directory:
  ```bash
  ANLP_PROJECT=/path/to/ANLP sbatch sbatches/01_data.sbatch
  ```

Logs go to `sbatches/logs/` with job ID in the filename.
