# ANLP: Topic Modeling for Song Lyrics

Compare topic modeling techniques with **OCTIS** on the [Genius Song Lyrics](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information/data) dataset (5M+ songs: title, tag, artist, year, lyrics). Use a **year-range subset** for algorithm comparison, then **BERTopic** with human-readable topic labels for:

- **Given a song** → get the most similar songs (by topic)
- **Given a topic** → get the most representative songs

## Setup

1. **Install** (from project root):

   ```bash
   uv sync
   # or: pip install -e .
   ```

2. **Dataset**: Download from Kaggle and place the CSV under `data/raw/`, or use Kaggle API:

   - Create `~/.kaggle/kaggle.json` with your API key, or set `KAGGLE_USERNAME` / `KAGGLE_KEY`
   - Run: `anlp data --year-min 2010 --year-max 2020 --max-docs 50000`  
     This will download (if needed) and build a cached subset.

   Dataset: [Genius Song Lyrics with Language Information](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information/data).

## Usage

### 1. Prepare data (subset by year range)

```bash
uv run anlp data --year-min 2010 --year-max 2020 --max-docs 50000
```

### 2. Compare OCTIS algorithms (LDA, NMF)

```bash
uv run anlp octis --year-min 2010 --year-max 2020 --max-docs 50000 --num-topics 20
```

Output: coherence (c-NPMI) and topic diversity per algorithm.

### 3. Fit BERTopic and save model (human-readable topics)

```bash
uv run anlp bertopic --year-min 2010 --year-max 2020 --max-docs 50000 --save models/bertopic_lyrics
```

Topics are represented with short phrases (n-gram range 1–2) for interpretability.

### 4. Similar songs for a given song

Uses the topic of the song and returns other songs in the same topic (by probability):

```bash
uv run anlp similar 0 --model models/bertopic_lyrics --top-k 10
```

`0` is the document index (row) in the processed corpus.

### 5. Most representative songs for a topic

```bash
uv run anlp representative 3 --model models/bertopic_lyrics --top-k 10
```

`3` is the topic ID (see BERTopic topic info for IDs and labels).

## Configuration

Edit `src/anlp/config.py` to change:

- **Data**: `YEAR_MIN`, `YEAR_MAX`, `MAX_DOCS_SUBSET`, `DATA_DIR`
- **OCTIS**: `OCTIS_ALGORITHMS`, `OCTIS_NUM_TOPICS`
- **BERTopic**: `BERTOPIC_EMBEDDING_MODEL`, `BERTOPIC_MIN_TOPIC_SIZE`, `BERTOPIC_REPRESENTATION_NGRAM_RANGE`, etc.
- **Retrieval**: `TOP_K_SIMILAR_SONGS`, `TOP_K_REPRESENTATIVE_SONGS`

## Project layout

- `src/anlp/config.py` – Paths and hyperparameters  
- `src/anlp/data/load_lyrics.py` – Load Kaggle CSV, filter by year, cache subset  
- `src/anlp/octis_compare.py` – OCTIS dataset build, LDA/NMF training, coherence & diversity  
- `src/anlp/bertopic_pipeline.py` – BERTopic fit with human-readable topic representation  
- `src/anlp/retrieval.py` – Similar songs (by song) and representative songs (by topic)  
- `src/anlp/cli.py` – CLI entrypoint (`anlp data|octis|bertopic|similar|representative`)

## Python API

```python
from anlp.data import load_lyrics_subset, get_lyrics_corpus
from anlp.octis_compare import compare_octis, build_octis_dataset, run_octis_model
from anlp.bertopic_pipeline import fit_bertopic_on_lyrics, get_topic_labels, load_bertopic_model
from anlp.retrieval import similar_songs_for_song, representative_songs_for_topic

# Load subset
df = load_lyrics_subset(year_min=2010, year_max=2020, max_docs=50_000)
corpus, df_clean = get_lyrics_corpus(df)

# Compare OCTIS
results = compare_octis(year_min=2010, year_max=2020, max_docs=50_000)

# BERTopic + retrieval
model, docs_df, topics, probs = fit_bertopic_on_lyrics(2010, 2020, 50_000)
similar = similar_songs_for_song(doc_id=0, docs_df=docs_df, topics=topics, probs=probs, top_k=10)
repr_songs = representative_songs_for_topic(topic_id=3, docs_df=docs_df, topics=topics, probs=probs, top_k=10)
```
