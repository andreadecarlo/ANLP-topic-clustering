"""Configuration for data paths, year range, and model parameters."""

from pathlib import Path

# Paths (relative to project root or env)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Kaggle dataset: Genius Song Lyrics with Language Information
# https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information/data
KAGGLE_DATASET = "carlosgdcj/genius-song-lyrics-with-language-information"
# Expected columns: title, tag, artist, year, lyrics (adapt if CSV differs)
# Zip from Kaggle extracts song_lyrics.csv
CSV_FILENAME = "song_lyrics.csv"

# Subset for comparing algorithms (full 5M+ is too large)
YEAR_MIN = 1960
YEAR_MAX = 1970
MAX_DOCS_SUBSET = 30_000  # cap for OCTIS comparison (optional)

# OCTIS comparison
OCTIS_ALGORITHMS = ["LDA", "NMF", "CTM"]  # LSI optional; CTM uses BERT embeddings (slower)
OCTIS_NUM_TOPICS = 20
OCTIS_RANDOM_STATE = 42

# BERTopic
BERTOPIC_NUM_TOPICS = "auto"  # or int
BERTOPIC_MIN_TOPIC_SIZE = 20
BERTOPIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast; use paraphrase-multilingual for non-English
BERTOPIC_REPRESENTATION_N_WORDS = 5
BERTOPIC_REPRESENTATION_NGRAM_RANGE = (1, 2)  # human-readable phrases
# Quantized Llama for topic labels (AutoModelForCausalLM + bitsandbytes). Set to None for c-TF-IDF only.
BERTOPIC_REPRESENTATION_LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"
# "cpu" = run Llama on CPU (avoids OOM when embeddings use GPU); "auto" = use GPU if available.
BERTOPIC_REPRESENTATION_LLAMA_DEVICE = "auto"

# Retrieval
TOP_K_SIMILAR_SONGS = 10
TOP_K_REPRESENTATIVE_SONGS = 10


def ensure_dirs() -> None:
    """Create data and model directories if they don't exist."""
    for d in (RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
