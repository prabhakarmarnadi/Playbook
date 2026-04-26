"""
LLM-based scenario summarizer for macro clustering.

Instead of concatenating field-labeled chunk excerpts, this module
prompts Azure OpenAI to generate a natural-language "scenario summary"
from the retrieved context. The resulting summaries are richer and more
semantically coherent, leading to better macro domain separation.

Falls back to the original field-summary approach if Azure OpenAI
is not configured (AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY).
"""
import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

logger = logging.getLogger(__name__)

SCENARIO_PROMPT = """You are a legal contract analyst. Given the following excerpts from a contract document, write a concise scenario summary (150-250 words) that describes:

1. What type of agreement this is (e.g., master services agreement, employment contract, NDA, lease, software license, etc.)
2. The key parties and their roles
3. The primary obligations and rights
4. Notable terms (payment, duration, termination, liability, IP, etc.)

Write in third person, factual tone. Focus on what makes this agreement distinctive.

DOCUMENT: {filename}

EXCERPTS:
{context}

SCENARIO SUMMARY:"""


def _get_client():
    """Create Azure OpenAI client from env vars. Returns None if not configured."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if not endpoint or not api_key:
        return None
    try:
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    except Exception as e:
        logger.warning(f"Failed to create Azure OpenAI client: {e}")
        return None


def _summarize_one(client, deployment: str, filename: str, context: str) -> str:
    """Generate a single scenario summary via Azure OpenAI."""
    prompt = SCENARIO_PROMPT.format(filename=filename, context=context)
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=400,
    )
    return response.choices[0].message.content.strip()


def generate_llm_summaries(
    agreements: list[dict],
    all_chunks: list[dict],
    chunk_embeddings: np.ndarray,
    embed_fn,
    max_summary_chars: int = 3000,
    max_workers: int = 5,
    progress_fn=None,
) -> list[str]:
    """
    Generate LLM scenario summaries for each agreement.

    For each document:
      1. Use FieldRegistry to select the most relevant chunk excerpts
      2. Send those excerpts to Azure OpenAI to produce a scenario summary
      3. Fall back to the field-summary text if the LLM call fails

    Args:
        agreements: List of agreement dicts with agreement_id, filename, raw_text.
        all_chunks: All chunks across all agreements.
        chunk_embeddings: (n_chunks, dim) embeddings.
        embed_fn: Embedding function for FieldRegistry.
        max_summary_chars: Max context chars sent to LLM per document.
        max_workers: Concurrent LLM calls.
        progress_fn: Optional progress(stage, msg) callback.

    Returns:
        List of summary strings, one per agreement.
    """
    from core.field_registry import FieldRegistry
    from config import SUMMARY_MAX_CHARS

    if max_summary_chars is None:
        max_summary_chars = SUMMARY_MAX_CHARS

    client = _get_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    if client is None:
        logger.warning("Azure OpenAI not configured — falling back to field-summary approach")
        from core.macro_clusterer import get_field_summaries
        return get_field_summaries(agreements, all_chunks, chunk_embeddings, embed_fn, max_summary_chars)

    # Build field-aware context for each doc (same as before)
    registry = FieldRegistry(embed_fn=embed_fn)
    registry.precompute_embeddings()

    agr_ids = [a["agreement_id"] for a in agreements]
    agr_chunk_indices: dict[str, list[int]] = {aid: [] for aid in agr_ids}
    for i, chunk in enumerate(all_chunks):
        aid = chunk.get("agreement_id", "")
        if aid in agr_chunk_indices:
            agr_chunk_indices[aid].append(i)

    # Pre-build the context strings
    contexts: list[tuple[str, str]] = []  # (filename, context_text)
    field_summaries: list[str] = []  # fallback if LLM fails

    for agr in agreements:
        aid = agr["agreement_id"]
        indices = agr_chunk_indices.get(aid, [])
        filename = agr.get("filename", "unknown")
        title = filename.replace(".pdf", "").replace(".txt", "").replace("_", " ")

        if not indices:
            text = agr.get("raw_text", "")[:2000]
            field_summaries.append(f"{title}\n\n{text}" if title else text)
            contexts.append((filename, text))
            continue

        doc_chunk_texts = [all_chunks[i]["text"] for i in indices]
        doc_chunk_embeddings = chunk_embeddings[indices]

        field_summary = registry.build_field_summary(
            doc_chunk_texts, doc_chunk_embeddings,
            top_k=2, min_score=0.25, max_summary_chars=max_summary_chars,
        )
        field_summaries.append(f"{title}\n\n{field_summary}" if title else field_summary)
        contexts.append((filename, field_summary))

    # Generate LLM summaries in parallel
    summaries = [None] * len(agreements)
    n_success = 0
    n_fallback = 0

    if progress_fn:
        progress_fn("llm_summary", f"Generating {len(agreements)} scenario summaries via {deployment}...")

    def _call_llm(idx: int) -> tuple[int, str | None]:
        fname, ctx = contexts[idx]
        try:
            return idx, _summarize_one(client, deployment, fname, ctx)
        except Exception as e:
            logger.warning(f"LLM summary failed for {fname}: {e}")
            return idx, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call_llm, i): i for i in range(len(agreements))}
        done_count = 0
        for future in as_completed(futures):
            idx, result = future.result()
            done_count += 1
            if result:
                summaries[idx] = result
                n_success += 1
            else:
                summaries[idx] = field_summaries[idx]
                n_fallback += 1
            if progress_fn and done_count % 50 == 0:
                progress_fn("llm_summary", f"  {done_count}/{len(agreements)} done")

    if progress_fn:
        progress_fn("llm_summary", f"Done: {n_success} LLM summaries, {n_fallback} fallbacks")

    logger.info(f"Generated {n_success} LLM summaries, {n_fallback} field-summary fallbacks")
    return summaries
