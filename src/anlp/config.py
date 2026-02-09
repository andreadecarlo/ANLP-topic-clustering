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
MAX_DOCS_SUBSET = 50_000  # cap for OCTIS comparison (optional)

# OCTIS comparison
OCTIS_ALGORITHMS = ["LDA", "NMF", "CTM"]  # LSI optional; CTM uses BERT embeddings (slower)
OCTIS_NUM_TOPICS = 20
OCTIS_RANDOM_STATE = 42

# BERTopic
# None = keep all clusters (fine-grained); "auto" = merge similar topics (can reduce to few); int = target count
# we cannot get more than 80 topics finetuned with llama2-7b-chat-hf
BERTOPIC_NUM_TOPICS = None
# Smaller = more, smaller topics (more fine-grained); larger = fewer, broader topics
BERTOPIC_MIN_TOPIC_SIZE = 15
BERTOPIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast; use paraphrase-multilingual for non-English
BERTOPIC_REPRESENTATION_N_WORDS = 5
BERTOPIC_REPRESENTATION_NGRAM_RANGE = (1, 3)  # human-readable phrases
# CountVectorizer for c-TF-IDF: min_df must be <= max_df; use 1 to avoid errors with small topics.
BERTOPIC_VECTORIZER_MIN_DF = 10
# Lighter model fits on GPU with embedding model. TinyLlama 1.1B 4-bit ~1–2 GB; Llama-2-7b needs device="cpu".
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BERTOPIC_REPRESENTATION_LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"
# "auto" = GPU (use with light model); "cpu" = CPU (slower, for Llama-2-7b if needed).
BERTOPIC_REPRESENTATION_LLAMA_DEVICE = "auto"
# Number of representative documents per topic in the [DOCUMENTS] section of the representation prompt.
BERTOPIC_REPRESENTATION_NR_DOCS = 16
# Truncate each doc in [DOCUMENTS] to this many words ("whitespace" tokenizer). None = no truncation.
BERTOPIC_REPRESENTATION_DOC_LENGTH = 100
# Set BERTOPIC_REPRESENTATION_LLAMA_MODEL = None to skip LLM and use c-TF-IDF only.

# BERTopic online (incremental) variant: partial_fit over chunks
# https://maartengr.github.io/BERTopic/getting_started/online/online.html
BERTOPIC_ONLINE_CHUNK_SIZE = 50_000  # documents per partial_fit chunk
BERTOPIC_ONLINE_N_COMPONENTS = 5  # IncrementalPCA dimensions
BERTOPIC_ONLINE_N_CLUSTERS = 15   # MiniBatchKMeans clusters (fixed number of topics)
BERTOPIC_ONLINE_DECAY = 0.1       # OnlineCountVectorizer: decay previous counts (0–1)
# Online vectorizer sees one row per topic (not per doc); use 1 so min_df <= nr of topics.
BERTOPIC_ONLINE_VECTORIZER_MIN_DF = 1

# BERTopic viz: batch size when re-embedding for reduced embeddings (online path)
BERTOPIC_VIZ_EMBED_BATCH_SIZE = 512

# Retrieval
TOP_K_SIMILAR_SONGS = 10
TOP_K_REPRESENTATIVE_SONGS = 10


def ensure_dirs() -> None:
    """Create data and model directories if they don't exist."""
    for d in (RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
