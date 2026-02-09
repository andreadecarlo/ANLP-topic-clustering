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

4. **BERTopic online (full dataset)** — incremental fit over chunks; use when the corpus is too large for a single `fit()`:
   ```bash
   sbatch sbatches/05_bertopic_online.sbatch
   ```
   Edit `--year-min`, `--year-max`, `--max-docs` in the script to subset; default saves to `models/bertopic_lyrics_online_full`.

5. **Visualization** — Generate visualizations from a saved model:
   
   **Using pre-trained model from Hugging Face Hub** (recommended for quick start):
   ```bash
   # Downloads model from Hugging Face and creates document map PNG + topic logs
   sbatch sbatches/04_visualize.sbatch
   ```
   
   - `04_visualize.sbatch`: Uses `scripts/load_visualize_save.py` to download the model from [Dr3dre/bertopic-lyrics-auto](https://huggingface.co/Dr3dre/bertopic-lyrics-auto), creates a document map PNG, and logs topics with representative songs. The model repository includes `docs.parquet` and `reduced_embeddings.npy`, so no additional data files are needed.
   
   **Using a local model** (after training with `03_bertopic.sbatch`):
   ```bash
   # Edit 04_visualize.sbatch to change --hf-repo-id to --model <local_path>
   # Or modify the script to use your local model path
   ```

## Visualization Details

### Output Files

- **`04_visualize.sbatch`** produces:
  - `models/bertopic_hf_document_map.png` - Static document map visualization
  - Topic logs printed to stdout (saved in log file)

### Customization

- **Change focus song**: Edit `--focus-title` in `04_visualize.sbatch` to highlight a specific song's topic
- **Change Hugging Face model**: Edit `HF_REPO_ID` variable in either script to use a different model
- **Change output location**: Edit `--output` or `--out-dir` parameters in the scripts
- **Adjust representative songs**: Edit `--n-representative` in `04_visualize.sbatch` to show more/fewer songs per topic

## Customisation

- **Partition / resources**: Edit the `#SBATCH` lines in each file (e.g. `--partition`, `--mem`, `--time`, `--cpus-per-task`).
- **Project path**: Set `ANLP_PROJECT` if you submit from another directory:
  ```bash
  ANLP_PROJECT=/path/to/ANLP sbatch sbatches/01_data.sbatch
  ```

Logs go to `sbatches/logs/` with job ID in the filename.
