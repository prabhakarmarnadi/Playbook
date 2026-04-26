"""
Micro clusterer — BERTopic within a single macro domain.

Legacy pain point: V1 runs one global BERTopic pass on field-summary embeddings.
This means "Payment Terms in SaaS" and "Payment Terms in Employment" get mixed,
and the embedding signal is averaged across fields, losing clause-level granularity.

V2 approach: Per-domain BERTopic on CLAUSE-LEVEL embeddings (FIELD-110).
Each chunk gets its own embedding. Clustering discovers clause types within a domain.
"""
import logging
import uuid

import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score
from umap import UMAP

from config import (
    MICRO_MIN_CLUSTER_SIZE, MICRO_MIN_SAMPLES,
    UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS, UMAP_MIN_DIST,
    LEGAL_STOPWORDS,
)
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Combined stopword set (English + legal domain)
COMBINED_STOPWORDS = set(ENGLISH_STOP_WORDS) | set(LEGAL_STOPWORDS)


class _StopwordFilteringAnalyzer:
    """Picklable analyzer filtering n-grams where ALL tokens are stopwords."""

    def __init__(self, ngram_range=(1, 4), stopwords=None):
        self.ngram_range = ngram_range
        self.stopwords = stopwords or COMBINED_STOPWORDS
        self._analyzer = None

    def _build_analyzer(self):
        if self._analyzer is None:
            self._analyzer = CountVectorizer(ngram_range=self.ngram_range).build_analyzer()
        return self._analyzer

    def __call__(self, doc):
        tokens = self._build_analyzer()(doc)
        return [t for t in tokens if any(w not in self.stopwords for w in t.split())]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_analyzer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def cluster_domain_chunks(
    chunk_texts: list[str],
    chunk_embeddings: np.ndarray,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> tuple[list[int], BERTopic]:
    """
    Run BERTopic on chunk-level embeddings within a single macro domain.

    Legacy equivalent: BERTopicClusterer.perform_clustering()
    V2 differences:
      - Runs per-domain (not globally)
      - Uses clause-level embeddings (not field-summary averages, solving FIELD-110)
      - Smaller cluster sizes (finer-grained clause types)

    Returns (topic_ids_per_chunk, topic_model).
    """
    n = len(chunk_texts)
    mcs = min_cluster_size or MICRO_MIN_CLUSTER_SIZE
    ms = min_samples or MICRO_MIN_SAMPLES

    # Adapt UMAP parameters to dataset size
    n_neighbors = min(UMAP_N_NEIGHBORS, n - 1)
    n_components = min(UMAP_N_COMPONENTS, n - 2, n_neighbors - 1)
    n_components = max(2, n_components)

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(mcs, 2),
        min_samples=max(ms, 1),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        analyzer=_StopwordFilteringAnalyzer(ngram_range=(1, 4), stopwords=COMBINED_STOPWORDS),
        max_features=100000,
        min_df=1,
    )
    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=True, reduce_frequent_words=True,
    )

    from core.cuml_wrappers import cuml_available
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=cuml_available(),
        verbose=False,
    )

    topics, _probs = topic_model.fit_transform(chunk_texts, chunk_embeddings)

    # Reduce outliers — lower threshold to aggressively reclaim chunks
    if -1 in topics:
        try:
            new_topics = topic_model.reduce_outliers(
                chunk_texts, topics, strategy="c-tf-idf", threshold=0.1
            )
            topic_model.update_topics(
                chunk_texts, topics=new_topics,
                vectorizer_model=topic_model.vectorizer_model,
                representation_model=None,
            )
            topics = new_topics
        except Exception:
            pass  # Outlier reduction is best-effort

    n_clusters = len(set(t for t in topics if t != -1))
    logger.info(f"  Micro clustering: {n_clusters} clause types from {n} chunks")
    return topics, topic_model


def compute_cluster_quality(
    chunk_embeddings: np.ndarray,
    topics: list[int],
) -> dict[int, float]:
    """
    Compute per-cluster quality scores.

    Legacy equivalent: Scores model (average_cosine_similarity, silhouette, DBCV).
    V2: Per-cluster cosine similarity to centroid (more granular than global metrics).
    """
    quality = {}
    unique_topics = set(t for t in topics if t != -1)

    for tid in unique_topics:
        mask = [i for i, t in enumerate(topics) if t == tid]
        if len(mask) < 2:
            quality[tid] = 1.0
            continue
        cluster_embs = chunk_embeddings[mask]
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        sims = cluster_embs @ centroid
        quality[tid] = float(np.mean(sims))

    return quality


def label_micro_clusters(llm: LLMClient, topic_model: BERTopic, domain_label: str) -> dict[int, dict]:
    """
    Label each micro cluster (clause type) with a single LLM call per cluster.

    Legacy equivalent: topic_namer.py with OpenAI representation model.
    V2 difference: Domain-aware prompt (the LLM knows this is SaaS, not NDA).
    """
    labels = {}
    topic_info = topic_model.get_topic_info()

    for _idx, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            labels[-1] = {"label": "Miscellaneous", "description": "Chunks not assigned to a clause type"}
            continue

        keywords = [w for w, _ in topic_model.get_topic(tid)]
        rep_docs = topic_model.get_representative_docs(tid)

        prompt = f"""You are labeling clusters of legal clauses found within {domain_label} agreements.

Keywords for this cluster: {', '.join(keywords[:10])}

Representative clause excerpts:
{chr(10).join(f'- {doc[:300]}' for doc in (rep_docs or [])[:3])}

What type of legal clause is this? Respond in JSON:
{{"label": "clause type name like Payment Terms, Liability Limitation, Auto-Renewal", "description": "one sentence describing what this clause covers"}}"""

        result = llm.complete_json(prompt)
        labels[tid] = result
        logger.info(f"    Clause {tid}: {result.get('label', '?')}")

    return labels


def build_cluster_records(
    domain_id: str,
    chunk_ids: list[str],
    topics: list[int],
    embeddings: np.ndarray,
    labels: dict[int, dict],
    quality_scores: dict[int, float],
    topic_model: BERTopic,
) -> list[dict]:
    """Build cluster records with centroids, keywords, and quality metrics."""
    records = []
    unique_topics = sorted(set(t for t in topics if t != -1))

    for tid in unique_topics:
        mask = [i for i, t in enumerate(topics) if t == tid]
        cluster_embs = embeddings[mask]
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        keywords = [w for w, _ in topic_model.get_topic(tid)][:15]
        info = labels.get(tid, {"label": f"Clause_{tid}", "description": ""})

        cluster_id = f"c_{uuid.uuid4().hex[:8]}"
        records.append({
            "cluster_id": cluster_id,
            "topic_id": tid,
            "domain_id": domain_id,
            "label": info.get("label", f"Clause_{tid}"),
            "description": info.get("description", ""),
            "keywords": keywords,
            "chunk_count": len(mask),
            "quality_score": quality_scores.get(tid, 0.0),
            "centroid": centroid,
            "chunk_indices": mask,
            "chunk_ids": [chunk_ids[i] for i in mask],
        })

    return records
