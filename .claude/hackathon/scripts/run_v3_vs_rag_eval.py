#!/usr/bin/env python3
"""
V3 Pipeline vs Hackathon RAG — Comprehensive Evaluation
============================================================

Runs the new minimal_pipeline's RLM V3 discovery + core.rag_extractor
extraction on benchmark clusters, then compares against the previous
best (hackathon RAG pipeline).

Measurements:
  1. Core: fill rate, verbatim rate, hallucination rate (vs PREV_RAG)
  2. Dedup effectiveness: fields before/after dedup, cross-doc value dedup
  3. Clause sub-types: discovered vs persisted, quality check
  4. Intent extraction: run on sample chunks, measure coverage
  5. Confidence gate: simulate gate pass/fail on extraction confidences

Previous best (from unified eval):
  - Discovery: DSPy CoT + trial extraction filter
  - Extraction: scripts/rag_extraction.py (hybrid retrieval + per-field LLM)
  - Result: 0% halluc, 98.3% fill, 94.3% verbatim, 17.5s avg

New pipeline (minimal_pipeline):
  - Discovery: RLM V3 (agentic REPL, 12 tools, dual-mode workers, Latent Briefing)
  - Extraction: core/rag_extractor.py (async RAG with agentic retry + reviewer + model routing)
"""
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)-25s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Benchmark clusters (same as unified eval) ────────────────────────────────

BENCHMARK_CLUSTERS_MACRO = [
    "ct_f778e156",   # Franchise agreement (392 chunks)
    "ct_89e477aa",   # Offering Allocation Compliance (860 chunks)
    "ct_e318cbba",   # Website Co-Branding Agreement (44 chunks)
    "ct_115cb3ec",   # Defined Terms and Exceptions (28 chunks)
    "ct_5a3f3343",   # Stock option grants (78 chunks)
]

BENCHMARK_CLUSTERS_MICRO = [
    "ct_8aa78fea",   # Currency denomination clause (10 chunks)
    "ct_4f0609ee",   # Trademark License Terms (10 chunks)
    "ct_38f33255",   # Shipping fee schedule (11 chunks)
    "ct_49261ddd",   # Merger stock restrictions (12 chunks)
    "ct_ac9514ba",   # Development Definitions (18 chunks)
]

# Default to macro
BENCHMARK_CLUSTERS = BENCHMARK_CLUSTERS_MACRO

DB_PATH = str(Path(__file__).parent.parent / "data" / "cuad_510_demo.duckdb")
N_EXTRACT_CHUNKS = 8  # Same as unified eval

# ── Discovery chunk sampling ──────────────────────────────────────────────────
# Sample a random percentage of chunks for discovery (avoids bias from first-N)
DISCOVERY_SAMPLE_FRACTION = 0.25  # 25% of cluster chunks
DISCOVERY_SAMPLE_MIN = 15         # floor: always at least 15 chunks
DISCOVERY_SAMPLE_MAX = 60         # ceiling: never more than 60 chunks

# ── Junk field suffixes to filter out after discovery ─────────────────────────
JUNK_FIELD_SUFFIXES = (
    "_heading", "_section", "_header", "_title_only",
    "_toc", "_table_of_contents", "_placeholder",
)

# ── Previous best results (from unified eval JSON) ───────────────────────────

PREV_RAG_RESULTS = {
    "ct_f778e156": {"fields": 15, "fill": 1.0, "verbatim": 0.867, "halluc": 0.0,
                    "disc_s": 4.49, "ext_s": 13.06, "retries": 0, "reviewer": 2},
    "ct_89e477aa": {"fields": 13, "fill": 1.0, "verbatim": 1.0, "halluc": 0.0,
                    "disc_s": 3.5, "ext_s": 10.71, "retries": 0, "reviewer": 0},
    "ct_e318cbba": {"fields": 13, "fill": 1.0, "verbatim": 0.846, "halluc": 0.0,
                    "disc_s": 12.74, "ext_s": 11.47, "retries": 0, "reviewer": 2},
    "ct_115cb3ec": {"fields": 12, "fill": 0.917, "verbatim": 1.0, "halluc": 0.0,
                    "disc_s": 6.0, "ext_s": 11.81, "retries": 3, "reviewer": 0},
    "ct_5a3f3343": {"fields": 13, "fill": 1.0, "verbatim": 1.0, "halluc": 0.0,
                    "disc_s": 3.89, "ext_s": 10.05, "retries": 0, "reviewer": 0},
}


# ── Grounding utilities ──────────────────────────────────────────────────────

def grounding_check(value: str, chunk_texts: list[str]) -> dict:
    """Check if extracted value is grounded in source chunks."""
    if not value:
        return {"level": "missing", "score": 0.0}
    v_lower = value.lower().strip()
    for ct in chunk_texts:
        if v_lower in ct.lower():
            return {"level": "verbatim", "score": 1.0}
    v_tokens = set(re.findall(r'\b\w+\b', v_lower))
    if not v_tokens:
        return {"level": "missing", "score": 0.0}
    all_tokens = set()
    for ct in chunk_texts:
        all_tokens.update(re.findall(r'\b\w+\b', ct.lower()))
    overlap = len(v_tokens & all_tokens) / len(v_tokens)
    if overlap >= 0.8:
        return {"level": "high_overlap", "score": 0.8}
    return {"level": "hallucinated", "score": 0.0}


# ── Run RLM V3 Discovery ─────────────────────────────────────────────────────

def _sample_discovery_chunks(all_chunks: list[str]) -> list[str]:
    """Randomly sample a fraction of cluster chunks for discovery.

    Uses DISCOVERY_SAMPLE_FRACTION of total, clamped between
    DISCOVERY_SAMPLE_MIN and DISCOVERY_SAMPLE_MAX.
    """
    n_total = len(all_chunks)
    n_sample = max(
        DISCOVERY_SAMPLE_MIN,
        min(DISCOVERY_SAMPLE_MAX, int(n_total * DISCOVERY_SAMPLE_FRACTION)),
    )
    # If cluster is smaller than the min, just use all chunks
    if n_total <= n_sample:
        return all_chunks
    return random.sample(all_chunks, n_sample)


def _filter_junk_fields(fields: list[dict]) -> list[dict]:
    """Remove non-extractable fields (section headings, TOC entries, etc.)."""
    filtered = []
    for f in fields:
        name = f.get("name", "").lower()
        if any(name.endswith(s) for s in JUNK_FIELD_SUFFIXES):
            continue
        filtered.append(f)
    return filtered


def run_rlm_v3_discovery(cluster_label, keywords, chunk_texts, lm=None, sub_lm=None):
    """Run RLM V3 discovery on a cluster's chunks."""
    from core.rlm_v3 import RLMV3, RLMV3Config

    config = RLMV3Config(
        max_iterations=15,
        max_llm_calls=30,
        max_sample_chunks=60,
        max_fields=25,
        max_time_s=300,  # 5 min wall-clock cap per cluster
        state_field_budget=30,
        state_prune_keep=25,
        token_budget=300_000,
    )
    v3 = RLMV3(config=config, lm=lm, sub_lm=sub_lm)
    return v3.discover(cluster_label, keywords, chunk_texts)


# ── Dedup Effectiveness Measurement ──────────────────────────────────────────

def measure_dedup_effectiveness(rlm_meta: dict, batch_results: list | None) -> dict:
    """Measure field dedup from RLM phase2 postprocessing and cross-doc value dedup.

    Args:
        rlm_meta: Metadata dict from RLM V3 discover().
        batch_results: List of ExtractionResult objects from RAG extraction.

    Returns:
        Dict with dedup metrics: field_dedup_rate, cross_doc_value_dupes, etc.
    """
    metrics: dict = {}

    # 1. Field-level dedup from RLM Phase 2
    p2 = rlm_meta.get("phase2_postprocess", {})
    fields_before = p2.get("fields_before", 0)
    fields_after = p2.get("fields_after", 0)
    fields_removed = p2.get("fields_removed", 0)
    metrics["field_dedup"] = {
        "before": fields_before,
        "after": fields_after,
        "removed": fields_removed,
        "dedup_rate": round(fields_removed / fields_before, 3) if fields_before else 0,
    }

    # 2. Cross-doc value dedup — check if different docs produce identical values
    if batch_results:
        from collections import defaultdict
        field_values: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for r in batch_results:
            if r.value and r.value.strip():
                field_values[r.field_name].append(
                    (r.agreement_id, r.value.strip().lower())
                )

        n_fields_multi_doc = 0
        n_fields_with_dupes = 0
        total_values = 0
        total_unique_values = 0
        for field_name, pairs in field_values.items():
            if len(pairs) > 1:
                n_fields_multi_doc += 1
                vals = [v for _, v in pairs]
                total_values += len(vals)
                unique = set(vals)
                total_unique_values += len(unique)
                if len(unique) < len(vals):
                    n_fields_with_dupes += 1

        metrics["cross_doc_value_dedup"] = {
            "fields_with_multi_doc": n_fields_multi_doc,
            "fields_with_identical_values": n_fields_with_dupes,
            "total_values": total_values,
            "unique_values": total_unique_values,
            "dedup_potential": round(
                1 - total_unique_values / total_values, 3
            ) if total_values > 0 else 0,
        }
    else:
        metrics["cross_doc_value_dedup"] = {}

    return metrics


# ── Clause Sub-type Measurement ──────────────────────────────────────────────

def measure_clause_subtypes(rlm_meta: dict, store, cluster_id: str) -> dict:
    """Measure clause sub-type discovery and persistence.

    Checks:
      - How many clause_types RLM V3 discovered
      - Whether they're persisted to DB (cluster_layers table)
      - Quality: avg description length, diversity of types

    Returns:
        Dict with clause sub-type metrics.
    """
    # From RLM meta
    clause_types = rlm_meta.get("clause_types", [])
    n_discovered = rlm_meta.get("n_clause_types", len(clause_types))

    # Check DB persistence in cluster_layers
    try:
        db_layers = store.conn.execute("""
            SELECT COUNT(*) as n_rows, COUNT(DISTINCT cluster_label) as n_labels
            FROM cluster_layers
            WHERE clustering_run LIKE ?
        """, [f"%{cluster_id}%"]).fetchone()
        n_persisted_rows = db_layers[0] if db_layers else 0
        n_persisted_labels = db_layers[1] if db_layers else 0
    except Exception:
        n_persisted_rows = 0
        n_persisted_labels = 0

    # Quality check on discovered clause types
    descriptions = [ct.get("description", "") for ct in clause_types
                    if isinstance(ct, dict)]
    avg_desc_len = (
        round(sum(len(d) for d in descriptions) / len(descriptions), 1)
        if descriptions else 0
    )
    names = [ct.get("name", "") for ct in clause_types if isinstance(ct, dict)]

    return {
        "n_discovered": n_discovered,
        "n_persisted_db_rows": n_persisted_rows,
        "n_persisted_db_labels": n_persisted_labels,
        "persisted": n_persisted_rows > 0,
        "names": names,
        "avg_description_length": avg_desc_len,
    }


# ── Intent Extraction Quality ────────────────────────────────────────────────

def measure_intent_quality(
    store, cluster_id: str, cluster_label: str,
    chunk_texts: list[str], n_sample: int = 5,
) -> dict:
    """Run intent extraction on a sample of chunks and measure quality.

    Uses the intent_extractor module to discover intents from sample chunks,
    then measures: coverage (% chunks with intents), diversity, and quality.
    """
    from core.intent_extractor import (
        IntentExtractionConfig,
        INTENT_DISCOVERY_PROMPT,
        _get_azure_client,
        _get_deployment,
        _call_azure,
        _normalize_intent,
    )

    config = IntentExtractionConfig(
        max_clause_chars=3000,
        max_retries=2,
    )
    client = _get_azure_client()
    deployment = _get_deployment(config)

    # Sample chunks for intent extraction
    sample = random.sample(chunk_texts, min(n_sample, len(chunk_texts)))

    total_intents = 0
    chunks_with_intents = 0
    all_labels: list[str] = []
    all_parties: set[str] = set()
    all_attributes: list[dict] = []

    for chunk_text in sample:
        prompt = INTENT_DISCOVERY_PROMPT.format(
            clause_type=cluster_label,
            clause_text=chunk_text[:config.max_clause_chars],
        )
        parsed = _call_azure(client, deployment, prompt, config)
        if not parsed or not isinstance(parsed, dict):
            continue

        raw_intents = parsed.get("intents", [])
        if not isinstance(raw_intents, list):
            continue

        chunk_found = 0
        for raw in raw_intents:
            intent = _normalize_intent(raw)
            if intent:
                chunk_found += 1
                all_labels.append(intent["label"])
                if intent["party_from"] != "none":
                    all_parties.add(intent["party_from"].lower())
                if intent["party_to"] != "none":
                    all_parties.add(intent["party_to"].lower())
                if intent["attributes"]:
                    all_attributes.append(intent["attributes"])

        if chunk_found > 0:
            chunks_with_intents += 1
        total_intents += chunk_found

    unique_labels = list(set(all_labels))
    return {
        "n_chunks_sampled": len(sample),
        "n_chunks_with_intents": chunks_with_intents,
        "coverage_rate": round(chunks_with_intents / len(sample), 3) if sample else 0,
        "total_intents": total_intents,
        "avg_intents_per_chunk": round(total_intents / len(sample), 2) if sample else 0,
        "unique_intent_labels": unique_labels,
        "n_unique_labels": len(unique_labels),
        "n_unique_parties": len(all_parties),
        "n_chunks_with_attributes": len(all_attributes),
    }


# ── Confidence Gate Impact ───────────────────────────────────────────────────

def measure_confidence_gate(batch_results: list | None, threshold: float = 0.6) -> dict:
    """Simulate confidence gate on extraction results.

    Computes what would happen if we applied a confidence gate before
    committing to full extraction. Measures the gate decision and the
    quality delta of gated vs ungated extraction.

    Args:
        batch_results: List of ExtractionResult objects.
        threshold: Confidence threshold for the gate.

    Returns:
        Dict with gate metrics: would_pass, mean_conf, impact on precision.
    """
    if not batch_results:
        return {
            "n_extractions": 0,
            "gate_threshold": threshold,
            "would_pass": False,
            "mean_confidence": 0.0,
        }

    confs = [r.confidence for r in batch_results
             if r.value and r.value.strip() and r.confidence is not None]
    if not confs:
        return {
            "n_extractions": len(batch_results),
            "gate_threshold": threshold,
            "would_pass": False,
            "mean_confidence": 0.0,
        }

    arr = np.array(confs)
    mean_conf = float(np.mean(arr))
    median_conf = float(np.median(arr))
    p25_conf = float(np.percentile(arr, 25))
    n_high = int(np.sum(arr >= 0.7))
    n_low = int(np.sum(arr < 0.4))

    # Gate decision
    gate_pass = mean_conf >= threshold

    # What if we filtered out low-confidence extractions?
    high_conf_results = [r for r in batch_results
                         if r.value and r.value.strip()
                         and r.confidence is not None and r.confidence >= 0.4]
    low_conf_results = [r for r in batch_results
                        if r.value and r.value.strip()
                        and r.confidence is not None and r.confidence < 0.4]

    return {
        "n_extractions": len(confs),
        "gate_threshold": threshold,
        "would_pass": gate_pass,
        "mean_confidence": round(mean_conf, 3),
        "median_confidence": round(median_conf, 3),
        "p25_confidence": round(p25_conf, 3),
        "n_high_conf": n_high,
        "n_low_conf": n_low,
        "pct_high_conf": round(n_high / len(confs), 3) if confs else 0,
        "n_after_filter": len(high_conf_results),
        "n_filtered_out": len(low_conf_results),
    }


# ── Run Core RAG Extraction ──────────────────────────────────────────────────

async def run_core_rag_extraction(store, cluster_id, fields, cluster_label, keywords):
    """Run the core.rag_extractor on a cluster."""
    from core.rag_extractor import RAGExtractor, RAGConfig
    from core.embedder import Embedder

    # Create async LLM client (same as minimal_pipeline uses)
    from openai import AzureOpenAI
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    deployment = "gpt-5.4"  # Use full model for extraction quality

    # Build a minimal AsyncLLMClient wrapper
    class MinimalLLM:
        """Thin wrapper matching the interface RAGExtractor expects."""
        def __init__(self):
            self.client = AzureOpenAI(
                azure_endpoint=endpoint, api_key=api_key, api_version=api_version,
            )
            self.deployment = deployment

        async def complete_json(self, prompt, temperature=0.0, max_tokens=1000, **kwargs):
            model = kwargs.get("model", self.deployment)
            loop = asyncio.get_event_loop()
            def _call():
                try:
                    resp = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                    return json.loads(resp.choices[0].message.content.strip())
                except Exception as e:
                    logger.warning(f"LLM error: {e}")
                    return None
            return await loop.run_in_executor(None, _call)

        async def complete(self, prompt, temperature=0.0, max_tokens=1000, **kwargs):
            model = kwargs.get("model", self.deployment)
            loop = asyncio.get_event_loop()
            def _call():
                try:
                    resp = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as e:
                    logger.warning(f"LLM error: {e}")
                    return None
            return await loop.run_in_executor(None, _call)

    llm = MinimalLLM()
    embedder = Embedder.get_instance()
    rag_config = RAGConfig(
        max_sample_docs=10,
        min_sample_docs=3,
        max_retry_rounds=3,
        enable_reviewer=True,
        min_extraction_confidence=0.4,
    )
    rag = RAGExtractor(
        store=store, llm=llm, embed_fn=embedder.embed, config=rag_config,
    )
    batch = await rag.extract_cluster(
        cluster_id=cluster_id,
        fields=fields,
        cluster_label=cluster_label,
        cluster_keywords=keywords,
    )
    return batch


# ── Main evaluation ──────────────────────────────────────────────────────────

async def run_eval():
    """Run V3 pipeline on all 5 benchmark clusters and compare."""
    import dspy
    from core.store import ClusteringStore

    # Configure DSPy LM (needed for RLM V3)
    # Main LM = gpt-5.4 for quality; sub LM = gpt-5.4-mini for cheaper sub-calls
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    lm = dspy.LM(
        model="azure/gpt-5.4",
        api_base=endpoint,
        api_key=api_key,
        api_version=api_version,
        max_tokens=8000,
        temperature=0.0,
    )
    sub_lm = dspy.LM(
        model="azure/gpt-5.4-mini",
        api_base=endpoint,
        api_key=api_key,
        api_version=api_version,
        max_tokens=8000,
        temperature=0.0,
    )
    dspy.configure(lm=lm)

    store = ClusteringStore(DB_PATH)
    results = {}

    for cluster_id in BENCHMARK_CLUSTERS:
        cluster = store.get_cluster(cluster_id)
        if not cluster:
            logger.warning(f"Cluster {cluster_id} not found")
            continue

        cluster_label = cluster.get("label", "Unknown")
        try:
            keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(
                cluster.get("keywords"), str) else (cluster.get("keywords") or [])
        except json.JSONDecodeError:
            keywords = []

        chunk_rows = store.get_chunks(cluster_id=cluster_id)
        all_chunk_texts = [r["chunk_text"] for r in chunk_rows]

        logger.info(f"\n{'='*80}")
        logger.info(f"CLUSTER: {cluster_label} ({cluster_id}) — {len(chunk_rows)} chunks")
        logger.info(f"{'='*80}")

        # ── Phase 1: RLM V3 Discovery ──
        logger.info("Phase 1: RLM V3 Discovery...")
        sample_texts = _sample_discovery_chunks(all_chunk_texts)
        logger.info(f"  Sampled {len(sample_texts)}/{len(all_chunk_texts)} chunks "
                    f"({len(sample_texts)/len(all_chunk_texts):.0%}) for discovery")
        t_disc = time.time()
        try:
            fields, rlm_meta = run_rlm_v3_discovery(
                cluster_label, keywords, sample_texts, lm=lm, sub_lm=sub_lm)
        except Exception as e:
            logger.error(f"RLM V3 discovery failed: {e}")
            fields, rlm_meta = [], {"error": str(e)}

        disc_time = round(time.time() - t_disc, 2)
        logger.info(f"  Discovery (raw): {len(fields)} fields in {disc_time}s")

        # Post-filter junk fields
        if fields:
            pre_filter = len(fields)
            fields = _filter_junk_fields(fields)
            if pre_filter != len(fields):
                logger.info(f"  Filtered: {pre_filter} → {len(fields)} fields "
                            f"(removed {pre_filter - len(fields)} junk fields)")
            logger.info(f"  Fields: {[f['name'] for f in fields]}")

        if not fields:
            logger.warning(f"  No fields discovered, skipping extraction")
            results[cluster_id] = {
                "cluster_label": cluster_label,
                "n_chunks": len(chunk_rows),
                "n_fields": 0,
                "discovery_time_s": disc_time,
                "rlm_meta": rlm_meta,
                "extractions": {},
                "metrics": {"fill": 0, "verbatim": 0, "halluc": 0, "grounded": 0},
                "dedup_metrics": measure_dedup_effectiveness(rlm_meta or {}, None),
                "clause_subtypes": measure_clause_subtypes(rlm_meta or {}, store, cluster_id),
                "intent_metrics": {},
                "confidence_gate": {},
            }
            continue

        # ── Phase 2: Core RAG Extraction ──
        logger.info("Phase 2: Core RAG Extraction...")
        t_ext = time.time()
        try:
            batch = await run_core_rag_extraction(
                store, cluster_id, fields, cluster_label, keywords)
            ext_time = round(time.time() - t_ext, 2)
            logger.info(f"  Extraction: {batch.meta.get('n_extractions', 0)} values in {ext_time}s")
            logger.info(f"  Retries: {batch.meta.get('total_retries', 0)}, "
                        f"Reviewer rejections: {batch.meta.get('total_reviewer_rejections', 0)}")
        except Exception as e:
            logger.error(f"RAG extraction failed: {e}")
            import traceback
            traceback.print_exc()
            ext_time = round(time.time() - t_ext, 2)
            batch = None

        # ── Compute metrics ──
        extractions = {}
        n_filled = 0
        n_verbatim = 0
        n_halluc = 0
        n_grounded = 0
        n_total_values = 0

        if batch and batch.results:
            for r in batch.results:
                if r.value is not None and r.value.strip():
                    n_filled += 1
                    n_total_values += 1
                    gnd = grounding_check(r.value, all_chunk_texts)
                    if gnd["level"] == "verbatim":
                        n_verbatim += 1
                        n_grounded += 1
                    elif gnd["level"] == "high_overlap":
                        n_grounded += 1
                    else:
                        n_halluc += 1
                    extractions[r.field_name] = {
                        "value": r.value,
                        "confidence": r.confidence,
                        "grounding": gnd["level"],
                        "model": r.model_used,
                        "retrieval_round": r.retrieval_round,
                        "reviewer": r.reviewer_accepted,
                    }

        # ── Fill rate: per-FIELD (matches PREV_RAG baseline metric) ──
        # A field is "filled" if ANY agreement produced a value for it.
        # This is fair: PREV_RAG fill was fields_with_value / total_fields.
        fields_with_value = set()
        if batch and batch.results:
            for r in batch.results:
                if r.value is not None and r.value.strip():
                    fields_with_value.add(r.field_name)
        fill_rate = len(fields_with_value) / len(fields) if fields else 0

        # Also compute per-(agreement, field) fill for thoroughness tracking
        n_agreements = batch.meta.get("n_agreements_sampled", 1) if batch else 1
        doc_field_fill = n_filled / (len(fields) * n_agreements) if (fields and n_agreements) else 0

        verbatim_rate = n_verbatim / n_total_values if n_total_values > 0 else 0
        halluc_rate = n_halluc / n_total_values if n_total_values > 0 else 0
        grounded_rate = n_grounded / n_total_values if n_total_values > 0 else 0

        metrics = {
            "fill": round(fill_rate, 3),
            "fill_doc_field": round(doc_field_fill, 3),
            "verbatim": round(verbatim_rate, 3),
            "halluc": round(halluc_rate, 3),
            "grounded": round(grounded_rate, 3),
            "n_fields_filled": len(fields_with_value),
            "n_filled": n_filled,
            "n_total_values": n_total_values,
            "n_agreements_sampled": n_agreements,
        }

        # ── Phase 3: Extended Measurements ──
        logger.info("Phase 3: Extended measurements...")

        # 3a. Dedup effectiveness
        dedup_metrics = measure_dedup_effectiveness(
            rlm_meta or {}, batch.results if batch else None)
        fd = dedup_metrics.get("field_dedup", {})
        logger.info(f"  Dedup: {fd.get('before',0)} → {fd.get('after',0)} fields "
                    f"({fd.get('dedup_rate',0):.0%} removed)")
        cdv = dedup_metrics.get("cross_doc_value_dedup", {})
        if cdv.get("total_values", 0) > 0:
            logger.info(f"  Cross-doc values: {cdv['total_values']} total, "
                        f"{cdv['unique_values']} unique "
                        f"({cdv['dedup_potential']:.0%} potential dedup)")

        # 3b. Clause sub-types
        clause_metrics = measure_clause_subtypes(rlm_meta or {}, store, cluster_id)
        logger.info(f"  Clause sub-types: {clause_metrics['n_discovered']} discovered"
                    f" (persisted={clause_metrics['persisted']})")
        if clause_metrics["names"]:
            logger.info(f"    Types: {clause_metrics['names'][:5]}"
                        f"{'...' if len(clause_metrics['names']) > 5 else ''}")

        # 3c. Intent extraction (sample of 5 chunks)
        logger.info("  Running intent extraction on 5 sample chunks...")
        t_intent = time.time()
        try:
            intent_metrics = measure_intent_quality(
                store, cluster_id, cluster_label, all_chunk_texts, n_sample=5)
            intent_time = round(time.time() - t_intent, 2)
            logger.info(f"  Intents: {intent_metrics['total_intents']} found in "
                        f"{intent_metrics['n_chunks_with_intents']}/{intent_metrics['n_chunks_sampled']} chunks "
                        f"({intent_metrics['coverage_rate']:.0%} coverage, {intent_time}s)")
            logger.info(f"    Labels: {intent_metrics['unique_intent_labels'][:5]}"
                        f"{'...' if intent_metrics['n_unique_labels'] > 5 else ''}")
        except Exception as e:
            logger.warning(f"  Intent extraction failed: {e}")
            intent_metrics = {"error": str(e)}
            intent_time = round(time.time() - t_intent, 2)

        # 3d. Confidence gate simulation
        gate_metrics = measure_confidence_gate(
            batch.results if batch else None, threshold=0.6)
        logger.info(f"  Confidence gate: would_pass={gate_metrics.get('would_pass', False)} "
                    f"(mean={gate_metrics.get('mean_confidence', 0):.2f}, "
                    f"high={gate_metrics.get('n_high_conf', 0)}, "
                    f"low={gate_metrics.get('n_low_conf', 0)})")

        results[cluster_id] = {
            "cluster_label": cluster_label,
            "n_chunks": len(chunk_rows),
            "n_fields": len(fields),
            "field_names": [f["name"] for f in fields],
            "discovery_time_s": disc_time,
            "extraction_time_s": ext_time,
            "total_time_s": round(disc_time + ext_time, 2),
            "rlm_meta": {k: v for k, v in (rlm_meta or {}).items()
                         if k not in ("fields", "extractions")},
            "rag_meta": batch.meta if batch else {},
            "extractions": extractions,
            "metrics": metrics,
            # Extended measurements
            "dedup_metrics": dedup_metrics,
            "clause_subtypes": clause_metrics,
            "intent_metrics": intent_metrics,
            "confidence_gate": gate_metrics,
        }

        logger.info(f"  Metrics: fill={metrics['fill']:.1%}, verbatim={metrics['verbatim']:.1%}, "
                     f"halluc={metrics['halluc']:.1%}")

    store.close()
    return results


def print_comparison(v3_results):
    """Print side-by-side comparison table."""
    has_baseline = any(cid in PREV_RAG_RESULTS for cid in BENCHMARK_CLUSTERS)
    mode = "MICRO" if not has_baseline else "MACRO"

    print("\n" + "=" * 100)
    print(f" V3 PIPELINE — {mode} CLUSTER EVALUATION" +
          (" vs HACKATHON RAG" if has_baseline else ""))
    print("=" * 100)

    # Aggregate
    v3_agg = {"fill": [], "verbatim": [], "halluc": [], "disc": [], "ext": [], "fields": []}
    rag_agg = {"fill": [], "verbatim": [], "halluc": [], "disc": [], "ext": [], "fields": []}

    header = f"{'Cluster':<35} │ {'Pipeline':<12} │ {'Fields':>6} │ {'Fill%':>6} │ {'Verb%':>6} │ {'Hall%':>6} │ {'Disc(s)':>7} │ {'Ext(s)':>7} │ {'Total':>7}"
    sep = "─" * len(header)
    print(f"\n{header}")
    print(sep)

    for cid in BENCHMARK_CLUSTERS:
        v3 = v3_results.get(cid, {})
        prev = PREV_RAG_RESULTS.get(cid, {})
        label = v3.get("cluster_label", cid)[:33]

        # V3 row
        v3m = v3.get("metrics", {})
        v3_disc = v3.get("discovery_time_s", 0)
        v3_ext = v3.get("extraction_time_s", 0)
        v3_total = v3.get("total_time_s", 0)
        v3_fields = v3.get("n_fields", 0)

        print(f"{label:<35} │ {'V3 NEW':<12} │ {v3_fields:>6} │ {v3m.get('fill',0):>5.1%} │ "
              f"{v3m.get('verbatim',0):>5.1%} │ {v3m.get('halluc',0):>5.1%} │ "
              f"{v3_disc:>7.1f} │ {v3_ext:>7.1f} │ {v3_total:>7.1f}")

        # Prev RAG row (only if baseline exists)
        if prev:
            prev_total = prev.get("disc_s", 0) + prev.get("ext_s", 0)
            print(f"{'':>35} │ {'Prev RAG':<12} │ {prev.get('fields',0):>6} │ {prev.get('fill',0):>5.1%} │ "
                  f"{prev.get('verbatim',0):>5.1%} │ {prev.get('halluc',0):>5.1%} │ "
                  f"{prev.get('disc_s',0):>7.1f} │ {prev.get('ext_s',0):>7.1f} │ {prev_total:>7.1f}")
        print(sep)

        # Aggregate
        v3_agg["fill"].append(v3m.get("fill", 0))
        v3_agg["verbatim"].append(v3m.get("verbatim", 0))
        v3_agg["halluc"].append(v3m.get("halluc", 0))
        v3_agg["disc"].append(v3_disc)
        v3_agg["ext"].append(v3_ext)
        v3_agg["fields"].append(v3_fields)

        if prev:
            rag_agg["fill"].append(prev.get("fill", 0))
            rag_agg["verbatim"].append(prev.get("verbatim", 0))
            rag_agg["halluc"].append(prev.get("halluc", 0))
            rag_agg["disc"].append(prev.get("disc_s", 0))
            rag_agg["ext"].append(prev.get("ext_s", 0))
            rag_agg["fields"].append(prev.get("fields", 0))

    # Averages
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print(f"\n{'AVERAGE':<35} │ {'V3 NEW':<12} │ {avg(v3_agg['fields']):>6.1f} │ {avg(v3_agg['fill']):>5.1%} │ "
          f"{avg(v3_agg['verbatim']):>5.1%} │ {avg(v3_agg['halluc']):>5.1%} │ "
          f"{avg(v3_agg['disc']):>7.1f} │ {avg(v3_agg['ext']):>7.1f} │ {avg(v3_agg['disc'])+avg(v3_agg['ext']):>7.1f}")

    if rag_agg["fill"]:  # Only show baseline comparison if we have data
        print(f"{'':>35} │ {'Prev RAG':<12} │ {avg(rag_agg['fields']):>6.1f} │ {avg(rag_agg['fill']):>5.1%} │ "
              f"{avg(rag_agg['verbatim']):>5.1%} │ {avg(rag_agg['halluc']):>5.1%} │ "
              f"{avg(rag_agg['disc']):>7.1f} │ {avg(rag_agg['ext']):>7.1f} │ {avg(rag_agg['disc'])+avg(rag_agg['ext']):>7.1f}")

        # Delta
        fill_delta = avg(v3_agg["fill"]) - avg(rag_agg["fill"])
        halluc_delta = avg(v3_agg["halluc"]) - avg(rag_agg["halluc"])
        speed_delta = (avg(v3_agg["disc"]) + avg(v3_agg["ext"])) - (avg(rag_agg["disc"]) + avg(rag_agg["ext"]))

        print(f"\n{'DELTA (V3 - Prev)':<35} │ {'':>12} │ {'':>6} │ {fill_delta:>+5.1%} │ "
              f"{'':>6} │ {halluc_delta:>+5.1%} │ {'':>7} │ {'':>7} │ {speed_delta:>+7.1f}")

        # Winner
        print("\n" + "=" * 100)
        if avg(v3_agg["halluc"]) <= avg(rag_agg["halluc"]) and avg(v3_agg["fill"]) >= avg(rag_agg["fill"]):
            print("WINNER: V3 Pipeline (better or equal on both hallucination AND fill rate)")
        elif avg(v3_agg["halluc"]) <= avg(rag_agg["halluc"]):
            print(f"WINNER on PRECISION: V3 Pipeline ({avg(v3_agg['halluc']):.1%} vs {avg(rag_agg['halluc']):.1%} halluc)")
            print(f"WINNER on RECALL: Prev RAG ({avg(rag_agg['fill']):.1%} vs {avg(v3_agg['fill']):.1%} fill)")
        elif avg(v3_agg["fill"]) >= avg(rag_agg["fill"]):
            print(f"WINNER on RECALL: V3 Pipeline ({avg(v3_agg['fill']):.1%} vs {avg(rag_agg['fill']):.1%} fill)")
            print(f"WINNER on PRECISION: Prev RAG ({avg(rag_agg['halluc']):.1%} vs {avg(v3_agg['halluc']):.1%} halluc)")
        else:
            print(f"WINNER: Prev RAG (better on both fill and hallucination)")
    else:
        # Micro mode — no baseline, just show V3 standalone results
        print("\n" + "=" * 100)
        print(f"MICRO CLUSTER RESULTS: {avg(v3_agg['fill']):.1%} fill, "
              f"{avg(v3_agg['verbatim']):.1%} verbatim, {avg(v3_agg['halluc']):.1%} halluc "
              f"({avg(v3_agg['fields']):.0f} fields avg)")

    print("=" * 100)

    # ── Extended Metrics Summary ──
    print_extended_metrics(v3_results)


def print_extended_metrics(v3_results):
    """Print the extended measurement results: dedup, clause types, intents, confidence gate."""
    print("\n" + "=" * 100)
    print(" EXTENDED MEASUREMENTS")
    print("=" * 100)

    # ── Dedup Effectiveness ──
    print("\n── DEDUP EFFECTIVENESS ──")
    print(f"  {'Cluster':<30} {'Before':>6} {'After':>6} {'Removed':>8} {'Rate':>6}  "
          f"{'XDoc-Vals':>9} {'Unique':>6} {'Dedup%':>6}")
    for cid in BENCHMARK_CLUSTERS:
        r = v3_results.get(cid, {})
        label = r.get("cluster_label", cid)[:28]
        dm = r.get("dedup_metrics", {})
        fd = dm.get("field_dedup", {})
        cdv = dm.get("cross_doc_value_dedup", {})
        print(f"  {label:<30} {fd.get('before',0):>6} {fd.get('after',0):>6} "
              f"{fd.get('removed',0):>8} {fd.get('dedup_rate',0):>5.0%}  "
              f"{cdv.get('total_values',0):>9} {cdv.get('unique_values',0):>6} "
              f"{cdv.get('dedup_potential',0):>5.0%}")

    # ── Clause Sub-types ──
    print("\n── CLAUSE SUB-TYPES ──")
    print(f"  {'Cluster':<30} {'Discovered':>10} {'Persisted':>10} {'AvgDesc':>8}  {'Types'}")
    for cid in BENCHMARK_CLUSTERS:
        r = v3_results.get(cid, {})
        label = r.get("cluster_label", cid)[:28]
        cm = r.get("clause_subtypes", {})
        names_str = ", ".join(cm.get("names", [])[:3])
        if len(cm.get("names", [])) > 3:
            names_str += "..."
        print(f"  {label:<30} {cm.get('n_discovered',0):>10} "
              f"{'Yes' if cm.get('persisted') else 'No':>10} "
              f"{cm.get('avg_description_length',0):>8.0f}  {names_str}")

    # ── Intent Extraction ──
    print("\n── INTENT EXTRACTION (5-chunk sample) ──")
    print(f"  {'Cluster':<30} {'Sampled':>7} {'w/Intent':>8} {'Coverage':>8} "
          f"{'Total':>6} {'Unique':>6} {'Avg/Chunk':>9}")
    for cid in BENCHMARK_CLUSTERS:
        r = v3_results.get(cid, {})
        label = r.get("cluster_label", cid)[:28]
        im = r.get("intent_metrics", {})
        if "error" in im or not im:
            print(f"  {label:<30} {'ERROR' if 'error' in im else 'N/A':>7}")
        else:
            print(f"  {label:<30} {im.get('n_chunks_sampled',0):>7} "
                  f"{im.get('n_chunks_with_intents',0):>8} "
                  f"{im.get('coverage_rate',0):>7.0%} "
                  f"{im.get('total_intents',0):>6} "
                  f"{im.get('n_unique_labels',0):>6} "
                  f"{im.get('avg_intents_per_chunk',0):>9.1f}")

    # ── Confidence Gate ──
    print("\n── CONFIDENCE GATE SIMULATION (threshold=0.6) ──")
    print(f"  {'Cluster':<30} {'Pass?':>5} {'Mean':>6} {'Median':>7} {'P25':>6} "
          f"{'High':>5} {'Low':>5} {'%High':>6}")
    for cid in BENCHMARK_CLUSTERS:
        r = v3_results.get(cid, {})
        label = r.get("cluster_label", cid)[:28]
        gm = r.get("confidence_gate", {})
        if not gm:
            print(f"  {label:<30} {'N/A':>5}")
        else:
            pass_str = "YES" if gm.get("would_pass") else "NO"
            print(f"  {label:<30} {pass_str:>5} "
                  f"{gm.get('mean_confidence',0):>5.2f} "
                  f"{gm.get('median_confidence',0):>6.2f} "
                  f"{gm.get('p25_confidence',0):>5.2f} "
                  f"{gm.get('n_high_conf',0):>5} "
                  f"{gm.get('n_low_conf',0):>5} "
                  f"{gm.get('pct_high_conf',0):>5.0%}")

    print("\n" + "=" * 100)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V3 Pipeline vs RAG Evaluation")
    parser.add_argument("--micro", action="store_true",
                        help="Evaluate on micro clusters (10-20 chunks) instead of macro")
    args = parser.parse_args()

    global BENCHMARK_CLUSTERS
    if args.micro:
        BENCHMARK_CLUSTERS = BENCHMARK_CLUSTERS_MICRO
        logger.info("Mode: MICRO clusters (10-20 chunks each)")
    else:
        BENCHMARK_CLUSTERS = BENCHMARK_CLUSTERS_MACRO
        logger.info("Mode: MACRO clusters (28-860 chunks each)")

    results = asyncio.run(run_eval())

    # Save results
    suffix = "_micro" if args.micro else ""
    out_path = Path(__file__).parent.parent / "data" / f"v3_vs_rag_comparison{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    # Print comparison
    print_comparison(results)


if __name__ == "__main__":
    main()
