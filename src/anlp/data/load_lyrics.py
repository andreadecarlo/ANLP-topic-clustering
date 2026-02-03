"""Load Kaggle lyrics CSV, filter by year, and build corpus for OCTIS/BERTopic."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from anlp.config import (
    CSV_FILENAME,
    KAGGLE_DATASET,
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
    max_docs: int = 50_000,
    csv_path: Path | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """
    Load lyrics CSV (from path or Kaggle), filter by year range, limit to max_docs.
    Reads in chunks and stops once enough rows are collected, so the full file
    is never loaded. Returns DataFrame with year, lyrics, and other subset columns.
    """
    path = _resolve_csv_path(csv_path)
    # Resolve which columns exist (CSV may use year/Year, lyrics/Lyrics/text)
    first = pd.read_csv(path, nrows=0)
    cols = list(first.columns)
    year_col = "year" if "year" in cols else ("Year" if "Year" in cols else None)
    lyrics_col = "lyrics" if "lyrics" in cols else ("Lyrics" if "Lyrics" in cols else ("text" if "text" in cols else None))
    if year_col is None or lyrics_col is None:
        raise ValueError(
            f"CSV must have year and lyrics (or Year, Lyrics/text). Found: {cols}"
        )
    usecols = [year_col, lyrics_col] + [
        c for c in _SUBSET_USECOLS if c in cols and c not in (year_col, lyrics_col)
    ]

    chunks: list[pd.DataFrame] = []
    n_seen = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.dropna(subset=[lyrics_col])
        chunk = chunk[(chunk[year_col] >= year_min) & (chunk[year_col] <= year_max)]
        if chunk.empty:
            continue
        chunks.append(chunk)
        n_seen += len(chunk)
        if n_seen >= max_docs:
            break

    if not chunks:
        return pd.DataFrame(columns=["year", "lyrics"])

    df = pd.concat(chunks, ignore_index=True)
    df = df.head(max_docs)
    # Normalize column names for downstream (lyrics, year)
    if lyrics_col != "lyrics":
        df = df.rename(columns={lyrics_col: "lyrics"})
    if year_col != "year":
        df = df.rename(columns={year_col: "year"})
    return df.reset_index(drop=True)


def get_lyrics_corpus(
    df: pd.DataFrame,
    min_chars: int = 100,
) -> tuple[list[str], pd.DataFrame]:
    """
    From a lyrics DataFrame, drop short/empty texts and return (list of doc strings, filtered df).
    """
    text_col = "lyrics" if "lyrics" in df.columns else "text"
    mask = df[text_col].astype(str).str.strip().str.len() >= min_chars
    df_clean = df.loc[mask].reset_index(drop=True)
    corpus = df_clean[text_col].astype(str).str.strip().tolist()
    return corpus, df_clean


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
