"""BERTopic pipeline with human-readable topic labels for song lyrics."""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
import transformers
from bertopic import BERTopic
from bertopic.representation import TextGeneration
from bertopic.vectorizers import OnlineCountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import CountVectorizer

from anlp.config import (
    BERTOPIC_EMBEDDING_MODEL,
    BERTOPIC_MIN_TOPIC_SIZE,
    BERTOPIC_NUM_TOPICS,
    BERTOPIC_ONLINE_CHUNK_SIZE,
    BERTOPIC_ONLINE_DECAY,
    BERTOPIC_ONLINE_N_CLUSTERS,
    BERTOPIC_ONLINE_N_COMPONENTS,
    BERTOPIC_REPRESENTATION_DOC_LENGTH,
    BERTOPIC_REPRESENTATION_LLAMA_DEVICE,
    BERTOPIC_REPRESENTATION_LLAMA_MODEL,
    BERTOPIC_REPRESENTATION_N_WORDS,
    BERTOPIC_REPRESENTATION_NGRAM_RANGE,
    BERTOPIC_REPRESENTATION_NR_DOCS,
    BERTOPIC_VECTORIZER_MIN_DF,
    BERTOPIC_VIZ_EMBED_BATCH_SIZE,
    MAX_DOCS_SUBSET,
    MODELS_DIR,
    ensure_dirs,
)
from anlp.data.load_lyrics import (
    get_lyrics_corpus,
    iter_corpus_chunks,
    load_lyrics_subset,
)

# Llama-2-style prompt for topic labeling: [KEYWORDS] and [DOCUMENTS] filled by BERTopic
LLAMA_REPRESENTATION_SYSTEM = """<s>[INST] <<SYS>>
You are a helpful assistant for labeling topics. Return only the topic label, nothing else.
<</SYS>>
"""

LLAMA_REPRESENTATION_MAIN = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information above, create a short label for this topic. Return only the label.
[/INST]
"""

# TinyLlama / Zephyr-style chat format (<|system|>, <|user|>, <|assistant|>)
TINYLLAMA_REPRESENTATION_PROMPT = """<|system|>
You are a helpful assistant for labeling topics. Return only the topic label, nothing else.
<|user|>
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information above, create a short label for this topic. Return only the label.
<|assistant|>
"""


def _make_llama_representation_model(
    model_id: str | None = BERTOPIC_REPRESENTATION_LLAMA_MODEL,
    device: str = BERTOPIC_REPRESENTATION_LLAMA_DEVICE,
    prompt: str | None = None,
    nr_docs: int = BERTOPIC_REPRESENTATION_NR_DOCS,
    doc_length: int | None = BERTOPIC_REPRESENTATION_DOC_LENGTH,
    tokenizer: str = "whitespace",
):
    """Build BERTopic representation with a quantized LLM (AutoModelForCausalLM + bitsandbytes).
    Use a light model (e.g. TinyLlama 1.1B) with device="auto" for GPU; Llama-2-7b needs device="cpu".
    doc_length truncates each document in [DOCUMENTS] (tokenizer: "whitespace" = words, "char" = chars).
    Set BERTOPIC_REPRESENTATION_LLAMA_MODEL to None to use c-TF-IDF only.
    """
    if not model_id:
        return None
    if prompt is None:
        # TinyLlama uses <|system|>/<|user|>/<|assistant|>; Llama-2 uses [INST]/[SYS]
        if "TinyLlama" in model_id:
            prompt = TINYLLAMA_REPRESENTATION_PROMPT
        else:
            prompt = LLAMA_REPRESENTATION_SYSTEM + LLAMA_REPRESENTATION_MAIN

    # Free memory before loading Llama (helps when embedding model already used GPU)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_cpu = (device or "").lower() == "cpu"
    device_map = "cpu" if use_cpu else "auto"
    # First load: download + 4-bit quantization can take several minutes; later runs use cache.
    print(f"Loading representation model '{model_id}' on {device_map} (first run may take a few min)...")
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Force pipeline to CPU when device is cpu so Llama never touches GPU (avoids OOM with embedding model on GPU)
    pipeline_device = -1 if use_cpu else 0  # -1 = CPU in HF pipeline
    # Greedy decoding (do_sample=False) avoids CUDA assert from bad logits in quantized models
    generator = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        max_new_tokens=50,
        repetition_penalty=1.1,
        do_sample=False,
        device=pipeline_device,
    )
    return TextGeneration(
        generator,
        prompt=prompt,
        nr_docs=nr_docs,
        doc_length=doc_length,
        tokenizer=tokenizer,
        pipeline_kwargs={"max_new_tokens": 50, "do_sample": False},
    )


def build_bertopic_model(
    documents: list[str],
    embedding_model: str = BERTOPIC_EMBEDDING_MODEL,
    min_topic_size: int = BERTOPIC_MIN_TOPIC_SIZE,
    nr_topics: str | int = BERTOPIC_NUM_TOPICS,
    n_gram_range: tuple[int, int] = BERTOPIC_REPRESENTATION_NGRAM_RANGE,
    n_words: int = BERTOPIC_REPRESENTATION_N_WORDS,
    representation_llama_model: str | None = BERTOPIC_REPRESENTATION_LLAMA_MODEL,
) -> BERTopic:
    """
    Build and fit BERTopic with human-readable topic representation.
    Uses n-gram range (1, 2) and top words; optionally fine-tunes labels via quantized Llama (AutoModelForCausalLM).
    """
    # Vectorizer for c-TF-IDF topic representation (phrases + single words)
    vectorizer = CountVectorizer(
        ngram_range=n_gram_range,
        stop_words="english",
        min_df=BERTOPIC_VECTORIZER_MIN_DF,
        max_df=1.0,
    )

    representation_model = _make_llama_representation_model(model_id=representation_llama_model)

    # Use dict so we can extract Llama labels and set them as topic names (tutorial: Topic_Modeling_with_Llama2.ipynb)
    if representation_model is not None:
        representation_model = {"Llama2": representation_model}

    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        verbose=True,
    )

    try:
        topic_model.fit(documents)
    except ValueError as e:
        if "Found array with 0 sample" in str(e):
            # Auto-reduction failed because all documents are outliers
            # Recreate model with auto-reduction disabled
            topic_model = BERTopic(
                embedding_model=embedding_model,
                min_topic_size=min_topic_size,
                nr_topics=None,  # Disable auto-reduction
                vectorizer_model=vectorizer,
                representation_model=representation_model,
                verbose=True,
            )
            topic_model.fit(documents)
        else:
            raise
    # Refine topic representation for human-readable labels (phrases)
    topic_model.update_topics(
        documents,
        n_gram_range=n_gram_range,
        top_n_words=n_words,
    )

    # Set Llama2 labels as the displayed topic names (follow Topic_Modeling_with_Llama2.ipynb)
    if representation_model is not None and "Llama2" in representation_model:
        try:
            full_repr = topic_model.get_topics(full=True)
            llama2_repr = full_repr.get("Llama2")
            main_repr = full_repr.get("Main")  # c-TF-IDF fallback when Llama returns empty/short
            if llama2_repr is not None:
                # set_topic_labels expects a list in sorted(unique_topics) order: -1, 0, 1, 2, ...
                topic_ids = sorted(llama2_repr.keys())
                llama2_labels = []
                for tid in topic_ids:
                    label = llama2_repr[tid]
                    text = (label[0][0].split("\n")[0] if label else "").strip()
                    # Fall back to c-TF-IDF when Llama returns empty or just a number (e.g. "1", "2")
                    if not text or text.isdigit():
                        if main_repr and tid in main_repr and main_repr[tid]:
                            text = " ".join(w[0] for w in main_repr[tid][:5])
                        else:
                            text = "Outliers" if tid == -1 else str(tid)
                    llama2_labels.append(text)
                topic_model.set_topic_labels(llama2_labels)
        except (KeyError, TypeError, IndexError):
            pass

    return topic_model


class _MiniBatchKMeansFloat64:
    """Wraps MiniBatchKMeans so X is cast to float64 before partial_fit (sklearn expects double; embeddings are often float32)."""

    def __init__(self, n_clusters: int = 50, random_state: int = 42):
        self._model = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=random_state
        )

    def partial_fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self._model.partial_fit(X, y)
        return self

    @property
    def labels_(self):
        return self._model.labels_


def build_bertopic_model_online(
    embedding_model: str = BERTOPIC_EMBEDDING_MODEL,
    n_components: int = BERTOPIC_ONLINE_N_COMPONENTS,
    n_clusters: int = BERTOPIC_ONLINE_N_CLUSTERS,
    decay: float = BERTOPIC_ONLINE_DECAY,
    n_gram_range: tuple[int, int] = BERTOPIC_REPRESENTATION_NGRAM_RANGE,
    representation_llama_model: str | None = BERTOPIC_REPRESENTATION_LLAMA_MODEL,
    random_state: int = 42,
) -> BERTopic:
    """
    Build BERTopic for online/incremental topic modeling (partial_fit with chunks).
    Uses IncrementalPCA, MiniBatchKMeans, and OnlineCountVectorizer per
    https://maartengr.github.io/BERTopic/getting_started/online/online.html
    Call partial_fit(docs) in a loop over document chunks; then set topic_model.topics_
    to the accumulated list. Representation (e.g. Llama) can be applied after the loop.
    For discovering new topics as data streams in, you can use the River package
    (e.g. cluster.DBSTREAM()) wrapped in a class with partial_fit and .labels_
    as hdbscan_model instead of MiniBatchKMeans; see the BERTopic online docs #river.
    """
    umap_model = IncrementalPCA(n_components=n_components)
    # Wrap so input is cast to float64; embedding pipeline often yields float32 and sklearn expects double
    cluster_model = _MiniBatchKMeansFloat64(
        n_clusters=n_clusters, random_state=random_state
    )
    vectorizer_model = OnlineCountVectorizer(
        ngram_range=n_gram_range,
        stop_words="english",
        decay=decay,
        min_df=BERTOPIC_VECTORIZER_MIN_DF,
        max_df=1.0,
    )
    representation_model = _make_llama_representation_model(
        model_id=representation_llama_model,
        device=BERTOPIC_REPRESENTATION_LLAMA_DEVICE,
        nr_docs=BERTOPIC_REPRESENTATION_NR_DOCS,
        doc_length=BERTOPIC_REPRESENTATION_DOC_LENGTH,
        tokenizer="whitespace",
    )
    if representation_model is not None:
        representation_model = {"Llama2": representation_model}

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=True,
    )
    return topic_model


def fit_bertopic_on_lyrics_online(
    year_min: int = 2010,
    year_max: int = 2020,
    max_docs: int | None = MAX_DOCS_SUBSET,
    chunk_size: int = BERTOPIC_ONLINE_CHUNK_SIZE,
    save_path: Path | None = None,
    refine_representations: bool = True,
) -> tuple[BERTopic, pd.DataFrame, list[int], list[float]]:
    """
    Load lyrics, fit BERTopic incrementally with partial_fit over chunks, then
    optionally refine topic representations (update_topics + Llama labels).
    Allows loading more data than fit_bertopic_on_lyrics by processing in chunks.
    """
    ensure_dirs()
    dataset = load_lyrics_subset(
        year_min=year_min, year_max=year_max, max_docs=max_docs or MAX_DOCS_SUBSET
    )
    corpus, dataset_clean = get_lyrics_corpus(dataset, min_chars=100)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    topic_model = build_bertopic_model_online()
    topics: list[int] = []
    n_chunks = 0
    for doc_chunk in iter_corpus_chunks(corpus, chunk_size):
        topic_model.partial_fit(doc_chunk)
        topics.extend(topic_model.topics_)
        n_chunks += 1
        if n_chunks % 10 == 0 or n_chunks == 1:
            print(f"  partial_fit chunk {n_chunks} ({len(doc_chunk)} docs)")

    topic_model.topics_ = topics

    if refine_representations:
        topic_model.update_topics(
            corpus,
            n_gram_range=BERTOPIC_REPRESENTATION_NGRAM_RANGE,
            top_n_words=BERTOPIC_REPRESENTATION_N_WORDS,
        )
        representation_model = getattr(
            topic_model, "representation_model", None
        ) or {}
        if isinstance(representation_model, dict) and "Llama2" in representation_model:
            try:
                full_repr = topic_model.get_topics(full=True)
                llama2_repr = full_repr.get("Llama2")
                main_repr = full_repr.get("Main")
                if llama2_repr is not None:
                    topic_ids = sorted(llama2_repr.keys())
                    llama2_labels = []
                    for tid in topic_ids:
                        label = llama2_repr[tid]
                        text = (
                            (label[0][0].split("\n")[0] if label else "").strip()
                        )
                        if not text or text.isdigit():
                            if (
                                main_repr
                                and tid in main_repr
                                and main_repr[tid]
                            ):
                                text = " ".join(
                                    w[0] for w in main_repr[tid][:5]
                                )
                            else:
                                text = "Outliers" if tid == -1 else str(tid)
                        llama2_labels.append(text)
                    topic_model.set_topic_labels(llama2_labels)
            except (KeyError, TypeError, IndexError):
                pass

    probs = getattr(topic_model, "probabilities_", None)
    if probs is None or len(probs) != len(topics):
        probs = [0.0] * len(topics)

    dataset_clean = dataset_clean.add_column("topic", topics)
    dataset_clean = dataset_clean.add_column("topic_prob", probs)

    if save_path is None:
        save_path = MODELS_DIR / "bertopic_lyrics_online"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    emb_model = BERTOPIC_EMBEDDING_MODEL
    if "/" not in emb_model:
        emb_model = f"sentence-transformers/{emb_model}"
    topic_model.save(
        str(save_path),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=emb_model,
    )
    # Free GPU memory before re-embedding for viz (reduces peak PyTorch allocation)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from anlp.bertopic_viz import compute_and_save_reduced_embeddings

    reduced_path = save_path.parent / (save_path.name + "_reduced_embeddings.npy")
    compute_and_save_reduced_embeddings(
        topic_model,
        corpus,
        reduced_path,
        embed_batch_size=BERTOPIC_VIZ_EMBED_BATCH_SIZE,
    )
    docs_path = save_path.parent / (save_path.name + "_docs.parquet")
    try:
        dataset_clean.to_parquet(str(docs_path))
    except (AttributeError, TypeError):
        dataset_clean.to_pandas().to_parquet(docs_path, index=False)

    df_clean = dataset_clean.to_pandas()
    return topic_model, df_clean, topics, list(probs) if hasattr(probs, "__iter__") else probs


def get_topic_labels(topic_model: BERTopic) -> dict[int, str]:
    """Get human-readable topic labels (Llama/custom if set via set_topic_labels, else c-TF-IDF names)."""
    info = topic_model.get_topic_info()
    # Prefer CustomName (Llama labels from set_topic_labels); Name is c-TF-IDF
    use_custom = "CustomName" in info.columns and info["CustomName"].notna().any()
    labels = {}
    for _, row in info.iterrows():
        tid = row["Topic"]
        if use_custom and "CustomName" in info.columns:
            name = row.get("CustomName", row["Name"])
        else:
            name = row["Name"]
        if tid == -1:
            labels[-1] = "Outliers" if pd.isna(name) or not str(name).strip() else str(name).strip()
        else:
            label = str(name).replace("_", " ").strip()
            if label.startswith("topic ") and len(label) > 6 and label[6:7].isdigit():
                label = " ".join(label.split()[2:])  # drop "topic N"
            labels[tid] = label or str(tid)
    return labels


def fit_bertopic_on_lyrics(
    year_min: int = 2010,
    year_max: int = 2020,
    max_docs: int | None = MAX_DOCS_SUBSET,
    save_path: Path | None = None,
) -> tuple[BERTopic, pd.DataFrame, list[int], list[float]]:
    """
    Load lyrics subset (Hugging Face Dataset), fit BERTopic, return model, metadata DataFrame, topics, and probs.
    Uses Dataset for loading/filtering to avoid loading the full CSV into memory.
    """
    ensure_dirs()
    dataset = load_lyrics_subset(
        year_min=year_min, year_max=year_max, max_docs=max_docs or MAX_DOCS_SUBSET
    )
    corpus, dataset_clean = get_lyrics_corpus(dataset, min_chars=100)

    # Free memory before fitting (helps when re-running or with many docs)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    topic_model = build_bertopic_model(corpus)
    topics, probs = topic_model.transform(corpus)

    dataset_clean = dataset_clean.add_column("topic", topics)
    dataset_clean = dataset_clean.add_column("topic_prob", probs)

    if save_path is None:
        save_path = MODELS_DIR / "bertopic_lyrics"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save with safetensors (light, safe); c-TF-IDF and embedding model pointer for reload
    emb_model = BERTOPIC_EMBEDDING_MODEL
    if "/" not in emb_model:
        emb_model = f"sentence-transformers/{emb_model}"
    topic_model.save(
        str(save_path),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=emb_model,
    )
    # Free GPU memory before re-embedding for viz (reduces peak PyTorch allocation)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save 2D reduced embeddings and docs parquet for visualize_documents / load_visualize_save
    from anlp.bertopic_viz import compute_and_save_reduced_embeddings

    reduced_path = save_path.parent / (save_path.name + "_reduced_embeddings.npy")
    compute_and_save_reduced_embeddings(
        topic_model,
        corpus,
        reduced_path,
        embed_batch_size=BERTOPIC_VIZ_EMBED_BATCH_SIZE,
    )
    docs_path = save_path.parent / (save_path.name + "_docs.parquet")
    try:
        dataset_clean.to_parquet(str(docs_path))
    except (AttributeError, TypeError):
        dataset_clean.to_pandas().to_parquet(docs_path, index=False)
    print(f"Saved model artifacts: {save_path}, {reduced_path.name}, {docs_path.name}")

    # Optional: run document map + barchart + topic map (notebook-style); do not interrupt on failure
    try:
        from anlp.bertopic_viz import load_reduced_embeddings, run_visualizations

        reduced = load_reduced_embeddings(reduced_path)
        out_dir = save_path.parent / (save_path.name + "_viz")
        df_clean = dataset_clean.to_pandas()
        titles = df_clean.get("title", df_clean.get("song", None))
        if titles is None:
            titles = [str(c)[:80] for c in corpus]
        else:
            titles = titles.tolist()
        doc_lengths = [len(c) for c in corpus]
        run_visualizations(
            topic_model,
            corpus,
            topics,
            reduced,
            out_dir,
            titles=titles,
            doc_lengths=doc_lengths,
        )
        print(f"Visualizations saved to {out_dir}")
    except Exception as e:
        import sys
        print(f"Visualization skipped (non-fatal): {e}", file=sys.stderr)

    # Return small DataFrame for CLI/retrieval compatibility
    df_clean = dataset_clean.to_pandas()
    return topic_model, df_clean, topics, probs


def load_bertopic_model(model_path: Path | str) -> BERTopic:
    """Load a saved BERTopic model."""
    return BERTopic.load(str(model_path))
