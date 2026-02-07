"""Load Kaggle lyrics CSV, filter by year, and build corpus for OCTIS/BERTopic."""

from __future__ import annotations

import re
from pathlib import Path

from datasets import Dataset

from anlp.config import (
    CSV_FILENAME,
    KAGGLE_DATASET,
    MAX_DOCS_SUBSET,
    RAW_DATA_DIR,
)


def _resolve_csv_path(csv_path: Path | None) -> Path:
    """Return path to lyrics CSV: use csv_path if given, else raw dir or Kaggle download."""
    if csv_path is not None and csv_path.exists():
        return Path(csv_path)
    candidate = RAW_DATA_DIR / CSV_FILENAME
    if candidate.exists():
        return candidate
    # Kaggle zip often extracts into a subdir named after the dataset
    for path in RAW_DATA_DIR.rglob(CSV_FILENAME):
        return path
    # Download from Kaggle
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import kaggle

        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(RAW_DATA_DIR),
            unzip=True,
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Lyrics CSV not found at {candidate}. "
            f"Download from https://www.kaggle.com/datasets/{KAGGLE_DATASET} "
            f"and place {CSV_FILENAME} in {RAW_DATA_DIR}, or pass --csv <path>. Kaggle download failed: {e}"
        ) from e
    for path in RAW_DATA_DIR.rglob(CSV_FILENAME):
        return path
    raise FileNotFoundError(
        f"After Kaggle download, {CSV_FILENAME} not found under {RAW_DATA_DIR}"
    )


# Columns to load when reading subset (avoids loading full CSV into memory)
_SUBSET_USECOLS = ["year", "lyrics", "title", "tag", "artist", "id"]


def load_lyrics_subset(
    year_min: int = 2010,
    year_max: int = 2020,
    max_docs: int = MAX_DOCS_SUBSET,
    csv_path: Path | None = None,
    chunksize: int = 100_000,
) -> Dataset:
    """
    Load lyrics CSV (from path or Kaggle), filter by year range, limit to max_docs.
    Reads in chunks and collects rows into a Hugging Face Dataset (no full CSV in memory).
    Returns Dataset with year, lyrics, and other subset columns (memory-mapped Arrow).
    """
    import pandas as pd

    path = _resolve_csv_path(csv_path)
    first = pd.read_csv(path, nrows=0)
    cols = list(first.columns)
    year_col = "year" if "year" in cols else ("Year" if "Year" in cols else None)
    lyrics_col = (
        "lyrics"
        if "lyrics" in cols
        else ("Lyrics" if "Lyrics" in cols else ("text" if "text" in cols else None))
    )
    if year_col is None or lyrics_col is None:
        raise ValueError(
            f"CSV must have year and lyrics (or Year, Lyrics/text). Found: {cols}"
        )
    usecols = [year_col, lyrics_col] + [
        c for c in _SUBSET_USECOLS if c in cols and c not in (year_col, lyrics_col)
    ]

    rows: list[dict] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.dropna(subset=[lyrics_col])
        chunk = chunk[
            (chunk[year_col] >= year_min) & (chunk[year_col] <= year_max)
        ]
        if chunk.empty:
            continue
        if lyrics_col != "lyrics":
            chunk = chunk.rename(columns={lyrics_col: "lyrics"})
        if year_col != "year":
            chunk = chunk.rename(columns={year_col: "year"})
        rows.extend(chunk.to_dict("records"))
        if len(rows) >= max_docs:
            break

    if not rows:
        return Dataset.from_dict({"year": [], "lyrics": []})

    rows = rows[:max_docs]
    return Dataset.from_list(rows)


def get_lyrics_corpus(
    dataset: Dataset,
    min_chars: int = 100,
) -> tuple[list[str], Dataset]:
    """
    From a lyrics Dataset, drop short/empty texts and return (list of doc strings, filtered Dataset).
    """
    text_col = "lyrics" if "lyrics" in dataset.column_names else "text"

    def has_min_chars(row):
        raw = row.get(text_col) or row.get("text") or ""
        return len(str(raw).strip()) >= min_chars

    filtered = dataset.filter(has_min_chars, batched=False)
    corpus = [str(s).strip() for s in filtered[text_col]]
    return corpus, filtered


def tokenize_for_octis(corpus: list[str]) -> list[list[str]]:
    """
    Tokenize documents for OCTIS: lowercase, keep alphabetic tokens only.
    Returns list of list of words per document.
    """
    tokenized = []
    for doc in corpus:
        words = re.findall(r"[a-z]+", doc.lower())
        tokenized.append(words)
    return tokenized
