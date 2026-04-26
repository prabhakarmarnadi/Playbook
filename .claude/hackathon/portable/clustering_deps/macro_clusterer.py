"""
Macro clusterer — NEW V2 capability (replaces flat BERTopic).

Legacy pain point: V1 runs a single BERTopic pass on field-summary embeddings.
All agreement types (NDA, SaaS, Employment) get clustered together, producing
muddled topics where "indemnity in an NDA" ≈ "indemnity in SaaS."

V2 approach: Two-stage clustering.
  Stage 1 (this module): Embed document summaries → BERTopic → macro domains.
  Stage 2 (micro_clusterer.py): Per-domain chunk-level BERTopic → clause types.

Key design decision: Uses embedding-based summary clustering, NOT per-doc LLM
classification. Per-doc LLM calls are 50-100x more expensive for 100-500 docs.
We make only 5-10 LLM calls to label the resulting macro clusters.
"""
import json
import logging
import uuid

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

from config import (
    MACRO_MIN_CLUSTER_SIZE, MACRO_MIN_SAMPLES, LEGAL_STOPWORDS,
)
from core.embedder import Embedder
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


def get_document_summaries(agreements: list[dict]) -> list[str]:
    """
    Build a summary string per document: filename + first ~500 tokens of body text.

    Legacy equivalent: AIFlow.call_field_summarization() — one LLM call per doc.
    V2: Simple truncation, 0 LLM calls. Summary clustering doesn't need paraphrasing.

    For field-aware summaries, use get_field_summaries() instead.
    """
    summaries = []
    for a in agreements:
        title = a.get("filename", "").replace(".pdf", "").replace(".txt", "").replace("_", " ")
        text = a.get("raw_text", "")[:2000]
        summaries.append(f"{title}\n\n{text}" if title else text)
    return summaries


def get_field_summaries(
    agreements: list[dict],
    all_chunks: list[dict],
    chunk_embeddings: np.ndarray,
    embed_fn,
    max_summary_chars: int | None = None,
) -> list[str]:
    """
    Build field-aware summaries per document using the legacy fields_processor approach.

    Instead of raw text truncation, this:
      1. Loads 55+ static field definitions (ported from 222 predefined field configs)
      2. Precomputes field key-term embeddings with Nomic/MiniLM
      3. Scores each document's chunks against all field embeddings (cosine similarity)
      4. Builds a summary from top-scoring chunks labeled with their field names

    Legacy: AIFlow.call_field_summarization() + PrecomputedEmbeddings + FieldExtractor
            (720+ fields, gRPC calls, 1 LLM call per doc for summary).
    V2: Same signal, pure vector math, 0 LLM calls.

    The result: macro clustering sees "[Cap On Liability] ... [Indemnification] ..."
    instead of raw boilerplate, producing cleaner domain separation.

    Args:
        agreements: List of agreement dicts with agreement_id.
        all_chunks: All chunks across all documents (with agreement_id key).
        chunk_embeddings: (n_chunks, dim) embeddings for all chunks.
        embed_fn: Embedding function for field key-terms.
        max_summary_chars: Max chars per summary. Defaults to config SUMMARY_MAX_CHARS.

    Returns:
        List of field-aware summary strings, one per agreement.
    """
    from core.field_registry import FieldRegistry
    from config import SUMMARY_MAX_CHARS

    if max_summary_chars is None:
        max_summary_chars = SUMMARY_MAX_CHARS

    registry = FieldRegistry(embed_fn=embed_fn)
    registry.precompute_embeddings()

    # Group chunks by agreement
    agr_ids = [a["agreement_id"] for a in agreements]
    agr_chunk_indices: dict[str, list[int]] = {aid: [] for aid in agr_ids}
    for i, chunk in enumerate(all_chunks):
        aid = chunk.get("agreement_id", "")
        if aid in agr_chunk_indices:
            agr_chunk_indices[aid].append(i)

    summaries = []
    for agr in agreements:
        aid = agr["agreement_id"]
        indices = agr_chunk_indices.get(aid, [])

        if not indices:
            # Fallback to raw text truncation
            title = agr.get("filename", "").replace(".pdf", "").replace(".txt", "").replace("_", " ")
            text = agr.get("raw_text", "")[:2000]
            summaries.append(f"{title}\n\n{text}" if title else text)
            continue

        doc_chunk_texts = [all_chunks[i]["text"] for i in indices]
        doc_chunk_embeddings = chunk_embeddings[indices]

        title = agr.get("filename", "").replace(".pdf", "").replace(".txt", "").replace("_", " ")
        field_summary = registry.build_field_summary(
            doc_chunk_texts, doc_chunk_embeddings,
            top_k=2, min_score=0.25, max_summary_chars=max_summary_chars,
        )

        summary = f"{title}\n\n{field_summary}" if title else field_summary
        summaries.append(summary)

    logger.info(f"Built {len(summaries)} field-aware summaries from {len(all_chunks)} chunks")
    return summaries


def cluster_into_domains(
    summaries: list[str],
    embedder: Embedder,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> tuple[list[int], np.ndarray, BERTopic]:
    """
    Run BERTopic on document-level summary embeddings to produce macro domains.

    Returns (topic_ids_per_doc, embeddings, topic_model).
    Typically produces 3-10 macro clusters from 100-500 docs.

    Legacy equivalent: Single BERTopic pass in bertopic_clusterer.py.
    V2 difference: This is the FIRST of two passes. Uses summary-level
    embeddings (1 per doc) instead of field-summary embeddings.
    """
    mcs = min_cluster_size or MACRO_MIN_CLUSTER_SIZE
    ms = min_samples or MACRO_MIN_SAMPLES

    embeddings = embedder.embed(summaries, show_progress=True)

    umap_model = UMAP(
        n_neighbors=min(15, len(summaries) - 1),
        n_components=min(5, len(summaries) - 2),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words=list(set(LEGAL_STOPWORDS)),
        ngram_range=(1, 2),
        min_df=2,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=False,
    )

    topics, _probs = topic_model.fit_transform(summaries, embeddings)
    logger.info(f"Macro clustering: {len(set(topics)) - (1 if -1 in topics else 0)} domains found from {len(summaries)} docs")

    return topics, embeddings, topic_model


def label_macro_clusters(llm: LLMClient, topic_model: BERTopic) -> dict[int, dict]:
    """
    Label each macro cluster with a single LLM call per cluster.

    With 5-10 clusters, this costs 5-10 LLM calls total (vs 100-500 for per-doc classification).

    Legacy equivalent: topic_namer.py OpenAI calls for topic naming.
    V2 difference: Returns structured JSON with label + description, not just a name.
    """
    labels = {}
    topic_info = topic_model.get_topic_info()

    for _idx, row in topic_info.iterrows():
        topic_id = row["Topic"]

        if topic_id == -1:
            labels[-1] = {"label": "Uncategorized", "description": "Documents not assigned to a domain"}
            continue

        keywords = [w for w, _ in topic_model.get_topic(topic_id)]
        rep_docs = topic_model.get_representative_docs(topic_id)

        prompt = f"""You are labeling a group of legal agreements that were automatically clustered together.

Keywords for this cluster: {', '.join(keywords[:10])}

Representative document excerpts (first 200 chars each):
{chr(10).join(f'- {doc[:200]}' for doc in (rep_docs or [])[:5])}

What type of agreement domain is this? Respond in JSON:
{{"label": "short name like SaaS, NDA, Employment, Vendor, Lease", "description": "one sentence describing this domain"}}"""

        result = llm.complete_json(prompt)
        labels[topic_id] = result
        logger.info(f"  Domain {topic_id}: {result.get('label', '?')}")

    return labels


def build_domain_records(
    agreements: list[dict],
    topics: list[int],
    embeddings: np.ndarray,
    labels: dict[int, dict],
) -> list[dict]:
    """
    Build domain records with centroids and per-agreement assignments.
    Returns list of domain dicts ready for store.upsert_domain().
    """
    domain_records = []
    unique_topics = sorted(set(t for t in topics if t != -1))

    for topic_id in unique_topics:
        mask = [i for i, t in enumerate(topics) if t == topic_id]
        centroid = embeddings[mask].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        info = labels.get(topic_id, {"label": f"Domain_{topic_id}", "description": ""})
        domain_id = f"d_{uuid.uuid4().hex[:8]}"

        domain_records.append({
            "domain_id": domain_id,
            "topic_id": topic_id,
            "label": info.get("label", f"Domain_{topic_id}"),
            "description": info.get("description", ""),
            "agreement_count": len(mask),
            "confidence": float(np.mean([
                float(np.dot(embeddings[i], centroid)) for i in mask
            ])),
            "centroid": centroid,
            "agreement_indices": mask,
        })

    # Handle outliers (topic -1)
    outlier_mask = [i for i, t in enumerate(topics) if t == -1]
    if outlier_mask:
        domain_records.append({
            "domain_id": "d_uncategorized",
            "topic_id": -1,
            "label": "Uncategorized",
            "description": "Documents not assigned to a domain",
            "agreement_count": len(outlier_mask),
            "confidence": 0.0,
            "centroid": embeddings[outlier_mask].mean(axis=0) if outlier_mask else np.zeros(embeddings.shape[1]),
            "agreement_indices": outlier_mask,
        })

    return domain_records
