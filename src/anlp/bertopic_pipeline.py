"""BERTopic pipeline with human-readable topic labels for song lyrics."""

from pathlib import Path

import pandas as pd
import torch
import transformers
from bertopic import BERTopic
from bertopic.representation import TextGeneration
from sklearn.feature_extraction.text import CountVectorizer

from anlp.config import (
    BERTOPIC_EMBEDDING_MODEL,
    BERTOPIC_MIN_TOPIC_SIZE,
    BERTOPIC_NUM_TOPICS,
    BERTOPIC_REPRESENTATION_LLAMA_DEVICE,
    BERTOPIC_REPRESENTATION_LLAMA_MODEL,
    BERTOPIC_REPRESENTATION_N_WORDS,
    BERTOPIC_REPRESENTATION_NGRAM_RANGE,
    MAX_DOCS_SUBSET,
    MODELS_DIR,
    ensure_dirs,
)
from anlp.data.load_lyrics import load_lyrics_subset, get_lyrics_corpus

# Llama-style prompt for topic labeling: [KEYWORDS] and [DOCUMENTS] filled by BERTopic (tutorial format)
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


def _make_llama_representation_model(
    model_id: str | None = BERTOPIC_REPRESENTATION_LLAMA_MODEL,
    device: str = BERTOPIC_REPRESENTATION_LLAMA_DEVICE,
    prompt: str | None = None,
    nr_docs: int = 4,
):
    """Build BERTopic representation with quantized Llama (AutoModelForCausalLM + bitsandbytes).
    See: https://maartengr.github.io/BERTopic/getting_started/representation/llm.html#llama-manual-quantization
    Set BERTOPIC_REPRESENTATION_LLAMA_MODEL to None to use c-TF-IDF only.
    device: "cpu" avoids GPU OOM when embeddings use GPU; "auto" uses GPU if available.
    """
    if not model_id:
        return None
    if prompt is None:
        prompt = LLAMA_REPRESENTATION_SYSTEM + LLAMA_REPRESENTATION_MAIN

    device_map = "cpu" if (device or "").lower() == "cpu" else "auto"
    # First load: download + 4-bit quantization can take several minutes; later runs use cache.
    print(f"Loading quantized Llama '{model_id}' on {device_map} (first run may take a few min)...")
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

    # Greedy decoding (do_sample=False) avoids CUDA assert from bad logits in quantized models
    generator = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        max_new_tokens=50,
        repetition_penalty=1.1,
        do_sample=False,
    )
    return TextGeneration(
        generator,
        prompt=prompt,
        nr_docs=nr_docs,
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
    # Use min_df=1 to avoid errors with small topics; max_df=1.0 allows all words
    vectorizer = CountVectorizer(
        ngram_range=n_gram_range,
        stop_words="english",
        min_df=1,  # Changed from 2 to avoid "max_df < min_df" error with small topics
        max_df=1.0,  # Changed from 0.95 to allow all words (no upper limit)
    )

    representation_model = _make_llama_representation_model(model_id=representation_llama_model)

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
    return topic_model


def get_topic_labels(topic_model: BERTopic) -> dict[int, str]:
    """Get human-readable topic labels (top terms or custom labels)."""
    info = topic_model.get_topic_info()
    labels = {}
    for _, row in info.iterrows():
        tid = row["Topic"]
        name = row["Name"]
        if tid == -1:
            labels[-1] = "Outliers"
        else:
            # Name is already "topic_0_word1_word2_..." – make it readable
            label = name.replace("_", " ").strip()
            if label.startswith("topic "):
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
    Load lyrics subset, fit BERTopic, return model, metadata DataFrame, topics, and probs.
    """
    ensure_dirs()
    df = load_lyrics_subset(year_min=year_min, year_max=year_max, max_docs=max_docs)
    corpus, df_clean = get_lyrics_corpus(df, min_chars=100)

    topic_model = build_bertopic_model(corpus)
    topics, probs = topic_model.transform(corpus)

    df_clean = df_clean.copy()
    df_clean["topic"] = topics
    df_clean["topic_prob"] = probs

    if save_path is None:
        save_path = MODELS_DIR / "bertopic_lyrics"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(save_path))
    # Save docs DataFrame (parquet preferred, fallback to CSV)
    docs_path = save_path.parent / (save_path.name + "_docs.parquet")
    try:
        df_clean.to_parquet(docs_path, index=False)
    except ImportError:
        # Fallback to CSV if parquet engine not available
        docs_path = save_path.parent / (save_path.name + "_docs.csv")
        df_clean.to_csv(docs_path, index=False)

    return topic_model, df_clean, topics, probs


def load_bertopic_model(model_path: Path | str) -> BERTopic:
    """Load a saved BERTopic model."""
    return BERTopic.load(str(model_path))
