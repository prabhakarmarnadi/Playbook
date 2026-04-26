#!/usr/bin/env python3
"""
Unified Eval Runner — All Pipelines on Same 5 Clusters
========================================================

Runs:
  1. RAG Extraction (DSPy discovery + RAG hybrid retrieval extraction)
  2. RLM Full (DSPy discovery + RLM extraction)

Then loads existing results for:
  3. Combined (DSPy discovery + Agentic extraction) — from data/combined_pipeline_comparison.json
  4. DSPy v2 only (discovery + extract_fields_from_chunk) — from data/dspy_v2_comparison.json
  5. OG baselines (standard_gpt, dspy_rlm, rlm_lb, rlm_v2) — from data/extraction_value_comparison_v2.json

And generates a unified comparison JSON + markdown report.
"""
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import dspy
from core.rlm_v2_dspy import (
    ClusterDiscoveryPipeline,
    _configure_dspy_lm,
    discovery_quality_metric,
)
from core.field_discovery import FieldDiscoveryConfig
from rag_extraction import RAGExtractionPipeline, RAGConfig, _ground_value as rag_ground
from rlm_full_pipeline import RLMFullPipeline, RLMFullConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def grounding_detail(value, chunk_text):
    """Compute grounding for a single value vs single chunk."""
    if not value or not chunk_text:
        return {"level": "missing", "score": 0.0}
    v_lower = value.lower().strip()
    c_lower = chunk_text.lower()
    if v_lower in c_lower:
        return {"level": "verbatim", "score": 1.0}
    v_tokens = set(re.findall(r'\b\w+\b', v_lower))
    c_tokens = set(re.findall(r'\b\w+\b', c_lower))
    if not v_tokens:
        return {"level": "empty", "score": 0.0}
    overlap = v_tokens & c_tokens
    ratio = len(overlap) / len(v_tokens)
    if ratio >= 0.8:
        return {"level": "high_overlap", "score": 0.8}
    return {"level": "hallucinated", "score": 0.0}


def grounding_detail_multi(value, chunk_texts):
    """Compute grounding for a value against multiple chunks (best match wins)."""
    if not value:
        return {"level": "missing", "score": 0.0}
    best = {"level": "hallucinated", "score": 0.0}
    for ct in chunk_texts:
        g = grounding_detail(value, ct)
        if g["score"] > best["score"]:
            best = g
    return best


def run_rag_pipeline(cluster_ids, conn, n_extract_chunks, embedder):
    """Run RAG extraction pipeline on all clusters."""
    logger.info("\n" + "█" * 80)
    logger.info(" RUNNING: RAG Extraction Pipeline")
    logger.info("█" * 80)

    from rag_extraction import _create_client as rag_create_client

    # Discovery pipeline (shared)
    disc_pipeline = ClusterDiscoveryPipeline(
        config=FieldDiscoveryConfig(), embedder=embedder
    )

    # RAG extraction pipeline
    rag_config = RAGConfig(
        top_k_chunks=n_extract_chunks,
        max_retries=3,
        enable_reviewer=True,
    )
    rag_pipeline = RAGExtractionPipeline(
        client=rag_create_client(),
        config=rag_config,
        embedder=embedder,
    )

    results = []
    for ci, cid in enumerate(cluster_ids):
        cluster = conn.execute(
            "SELECT cluster_id, label, keywords, chunk_count FROM clusters WHERE cluster_id = ?",
            [cid],
        ).fetchdf().to_dict("records")[0]

        cluster_label = cluster["label"] or "Unknown"
        try:
            keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(cluster.get("keywords"), str) else (cluster.get("keywords") or [])
        except json.JSONDecodeError:
            keywords = []

        chunks = conn.execute("""
            SELECT ch.chunk_text FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ca.cluster_id = ?
            ORDER BY ch.chunk_index
        """, [cid]).fetchdf()["chunk_text"].tolist()

        extract_chunks = chunks[:n_extract_chunks]

        logger.info(f"\n{'─'*60}")
        logger.info(f"[RAG {ci+1}/{len(cluster_ids)}] {cluster_label} ({len(chunks)} chunks)")

        # Phase 1: Discovery
        t0 = time.time()
        try:
            disc_result = disc_pipeline(
                cluster_label=cluster_label,
                keywords=", ".join(keywords[:10]),
                chunk_texts=chunks,
            )
            fields = disc_result.fields
            disc_meta = disc_result.meta
        except Exception as e:
            logger.error(f"  Discovery failed: {e}")
            fields = []
            disc_meta = {"error": str(e)}
        disc_time = round(time.time() - t0, 2)

        # Phase 2: RAG Extraction
        t1 = time.time()
        try:
            ext_result = rag_pipeline.extract(
                fields=fields,
                chunk_texts=extract_chunks,
                cluster_label=cluster_label,
            )
            ext_meta = ext_result.meta
        except Exception as e:
            logger.error(f"  RAG extraction failed: {e}")
            ext_result = None
            ext_meta = {"error": str(e)}
        ext_time = round(time.time() - t1, 2)

        # Compute comparison metrics
        n_fields = len(fields)
        total_slots = n_fields * len(extract_chunks)
        filled_slots = 0
        verbatim_count = 0
        halluc_count = 0
        grounded_count = 0

        extractions_for_report = {}
        if ext_result:
            for fname, ext in ext_result.extractions.items():
                if ext.value is not None:
                    g = grounding_detail_multi(ext.value, extract_chunks)
                    extractions_for_report[fname] = {
                        "value": ext.value,
                        "confidence": ext.confidence,
                        "grounding": g,
                        "retrieval_method": ext.retrieval_method,
                        "llm_tier": ext.llm_tier,
                        "n_retries": ext.n_retries,
                        "reviewer_verdict": ext.reviewer_verdict,
                    }
                    # Count for per-chunk metrics (field is "filled" in its source chunks)
                    for chunk_idx in ext.source_chunk_indices:
                        if chunk_idx < len(extract_chunks):
                            filled_slots += 1
                            if g["level"] == "verbatim":
                                verbatim_count += 1
                                grounded_count += 1
                            elif g["level"] == "high_overlap":
                                grounded_count += 1
                            elif g["level"] == "hallucinated":
                                halluc_count += 1
                            break  # count once per field
                else:
                    extractions_for_report[fname] = {
                        "value": None, "confidence": 0.0,
                        "grounding": {"level": "missing", "score": 0.0},
                    }

        # Document-level metrics (from compiled extractions)
        non_null = [e for e in extractions_for_report.values() if e.get("value") is not None]
        n_doc_extracted = len(non_null)
        n_doc_verbatim = sum(1 for e in non_null if e["grounding"]["level"] == "verbatim")
        n_doc_halluc = sum(1 for e in non_null if e["grounding"]["level"] == "hallucinated")

        agg = {
            "fill_rate": round(n_doc_extracted / n_fields, 3) if n_fields else 0,
            "verbatim_rate": round(n_doc_verbatim / n_doc_extracted, 3) if n_doc_extracted else 0,
            "grounded_rate": round((n_doc_extracted - n_doc_halluc) / n_doc_extracted, 3) if n_doc_extracted else 0,
            "hallucination_rate": round(n_doc_halluc / n_doc_extracted, 3) if n_doc_extracted else 0,
            "n_extracted": n_doc_extracted,
            "n_fields": n_fields,
        }

        results.append({
            "cluster_id": cid,
            "cluster_label": cluster_label,
            "chunk_count": cluster["chunk_count"],
            "n_fields": n_fields,
            "discovery_time_s": disc_time,
            "extraction_time_s": ext_time,
            "aggregate": agg,
            "disc_meta": disc_meta,
            "ext_meta": ext_meta,
            "extractions": extractions_for_report,
            "fields": fields,
        })

        logger.info(
            f"  fields={n_fields}  fill={agg['fill_rate']:.3f}  "
            f"verb={agg['verbatim_rate']:.3f}  halluc={agg['hallucination_rate']:.3f}  "
            f"disc={disc_time}s  ext={ext_time}s"
        )

    return results


def run_rlm_full_pipeline(cluster_ids, conn, n_extract_chunks, embedder):
    """Run RLM Full pipeline on all clusters."""
    logger.info("\n" + "█" * 80)
    logger.info(" RUNNING: RLM Full Pipeline (RLM Discovery + RLM Extraction)")
    logger.info("█" * 80)

    config = RLMFullConfig(
        max_chunks_for_extraction=n_extract_chunks,
        rlm_max_iterations=30,
        rlm_max_llm_calls=60,
    )
    pipeline = RLMFullPipeline(config=config, embedder=embedder)

    results = []
    for ci, cid in enumerate(cluster_ids):
        cluster = conn.execute(
            "SELECT cluster_id, label, keywords, chunk_count FROM clusters WHERE cluster_id = ?",
            [cid],
        ).fetchdf().to_dict("records")[0]

        cluster_label = cluster["label"] or "Unknown"
        try:
            keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(cluster.get("keywords"), str) else (cluster.get("keywords") or [])
        except json.JSONDecodeError:
            keywords = []

        chunks = conn.execute("""
            SELECT ch.chunk_text FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ca.cluster_id = ?
            ORDER BY ch.chunk_index
        """, [cid]).fetchdf()["chunk_text"].tolist()

        logger.info(f"\n{'─'*60}")
        logger.info(f"[RLM Full {ci+1}/{len(cluster_ids)}] {cluster_label} ({len(chunks)} chunks)")

        result = pipeline.run(
            cluster_label=cluster_label,
            keywords=", ".join(keywords[:10]),
            chunk_texts=chunks,
            n_extract_chunks=n_extract_chunks,
        )

        # Build comparison-compatible metrics
        fields = result["fields"]
        extractions = result["extractions"]
        ext_meta = result["ext_meta"]
        n_fields = len(fields)

        extract_chunks = chunks[:n_extract_chunks]
        non_null = {k: v for k, v in extractions.items() if v.get("value") is not None}
        n_extracted = len(non_null)
        n_verbatim = sum(1 for v in non_null.values() if v["grounding_level"] == "verbatim")
        n_halluc = sum(1 for v in non_null.values() if v["grounding_level"] == "hallucinated")

        agg = {
            "fill_rate": round(n_extracted / n_fields, 3) if n_fields else 0,
            "verbatim_rate": round(n_verbatim / n_extracted, 3) if n_extracted else 0,
            "grounded_rate": round((n_extracted - n_halluc) / n_extracted, 3) if n_extracted else 0,
            "hallucination_rate": round(n_halluc / n_extracted, 3) if n_extracted else 0,
            "n_extracted": n_extracted,
            "n_fields": n_fields,
        }

        results.append({
            "cluster_id": cid,
            "cluster_label": cluster_label,
            "chunk_count": cluster["chunk_count"],
            "n_fields": n_fields,
            "discovery_time_s": result["discovery_time_s"],
            "extraction_time_s": result["extraction_time_s"],
            "aggregate": agg,
            "disc_meta": result["disc_meta"],
            "ext_meta": ext_meta,
            "extractions": extractions,
            "fields": fields,
        })

        logger.info(
            f"  fields={n_fields}  fill={agg['fill_rate']:.3f}  "
            f"verb={agg['verbatim_rate']:.3f}  halluc={agg['hallucination_rate']:.3f}  "
            f"disc={result['discovery_time_s']}s  ext={result['extraction_time_s']}s"
        )

    return results


def load_existing_results():
    """Load previously computed results for Combined, DSPy v2, and baselines."""
    existing = {}

    # Combined pipeline
    combined_path = Path("data/combined_pipeline_comparison.json")
    if combined_path.exists():
        with open(combined_path) as f:
            data = json.load(f)
        existing["combined"] = data.get("combined_results", [])
        logger.info(f"Loaded Combined results: {len(existing['combined'])} clusters")
    else:
        logger.warning("Combined pipeline results not found — will skip")
        existing["combined"] = []

    # DSPy v2 only
    dspy_path = Path("data/dspy_v2_comparison.json")
    if dspy_path.exists():
        with open(dspy_path) as f:
            data = json.load(f)
        existing["dspy_v2"] = data.get("dspy_results", [])
        logger.info(f"Loaded DSPy v2 results: {len(existing['dspy_v2'])} clusters")
    else:
        logger.warning("DSPy v2 results not found — will skip")
        existing["dspy_v2"] = []

    # OG baselines
    baseline_path = Path("data/extraction_value_comparison_v2.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            data = json.load(f)
        existing["baselines"] = data.get("per_cluster", [])
        logger.info(f"Loaded OG baselines: {len(existing['baselines'])} clusters")
    else:
        logger.warning("OG baselines not found — will skip")
        existing["baselines"] = []

    return existing


def generate_eval_markdown(all_results: dict, output_path: str):
    """Generate comprehensive markdown evaluation report."""
    lines = []
    lines.append("# Unified Pipeline Evaluation Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("Comparison of all extraction pipelines on the same 5 clusters.")
    lines.append("")
    lines.append("### Pipelines Compared")
    lines.append("")
    lines.append("| # | Pipeline | Discovery | Extraction | Key Feature |")
    lines.append("|---|----------|-----------|------------|-------------|")
    lines.append("| 1 | **Combined** | DSPy CoT + trial filter | Agentic (carryforward + judge + reviewer) | Sequential context |")
    lines.append("| 2 | **RAG** | DSPy CoT + trial filter | Hybrid retrieval (RRF) + per-field LLM | Per-field retrieval |")
    lines.append("| 3 | **RLM Full** | DSPy CoT + trial filter | DSPy RLM (Python REPL) | Programmatic extraction |")
    lines.append("| 4 | **DSPy v2** | DSPy CoT + trial filter | extract_fields_from_chunk | Standard single-call |")
    lines.append("| 5 | **Standard GPT** | Template-based | Template extraction | Baseline |")
    lines.append("| 6 | **DSPy RLM (OG)** | DSPy RLM | Template extraction | Original RLM discovery |")
    lines.append("")

    # Aggregate table
    lines.append("## Aggregate Results (Across All Clusters)")
    lines.append("")
    lines.append("| Pipeline | Halluc% | Grounded% | Verbatim% | Fill% | Avg Disc(s) | Avg Ext(s) |")
    lines.append("|----------|---------|-----------|-----------|-------|-------------|------------|")

    pipeline_aggs = {}

    # RAG results
    if all_results.get("rag"):
        rag = all_results["rag"]
        n = len(rag)
        avg_halluc = sum(r["aggregate"]["hallucination_rate"] for r in rag) / n
        avg_ground = sum(r["aggregate"]["grounded_rate"] for r in rag) / n
        avg_verb = sum(r["aggregate"]["verbatim_rate"] for r in rag) / n
        avg_fill = sum(r["aggregate"]["fill_rate"] for r in rag) / n
        avg_disc = sum(r["discovery_time_s"] for r in rag) / n
        avg_ext = sum(r["extraction_time_s"] for r in rag) / n
        lines.append(f"| RAG | {avg_halluc*100:.1f}% | {avg_ground*100:.1f}% | {avg_verb*100:.1f}% | {avg_fill*100:.1f}% | {avg_disc:.1f} | {avg_ext:.1f} |")
        pipeline_aggs["rag"] = {"halluc": avg_halluc, "ground": avg_ground, "verb": avg_verb, "fill": avg_fill}

    # RLM Full results
    if all_results.get("rlm_full"):
        rlm = all_results["rlm_full"]
        n = len(rlm)
        avg_halluc = sum(r["aggregate"]["hallucination_rate"] for r in rlm) / n
        avg_ground = sum(r["aggregate"]["grounded_rate"] for r in rlm) / n
        avg_verb = sum(r["aggregate"]["verbatim_rate"] for r in rlm) / n
        avg_fill = sum(r["aggregate"]["fill_rate"] for r in rlm) / n
        avg_disc = sum(r["discovery_time_s"] for r in rlm) / n
        avg_ext = sum(r["extraction_time_s"] for r in rlm) / n
        lines.append(f"| RLM Full | {avg_halluc*100:.1f}% | {avg_ground*100:.1f}% | {avg_verb*100:.1f}% | {avg_fill*100:.1f}% | {avg_disc:.1f} | {avg_ext:.1f} |")
        pipeline_aggs["rlm_full"] = {"halluc": avg_halluc, "ground": avg_ground, "verb": avg_verb, "fill": avg_fill}

    # Combined (existing)
    if all_results.get("combined"):
        comb = all_results["combined"]
        n = len(comb)
        avg_halluc = sum(r["aggregate"]["hallucination_rate"] for r in comb) / n
        avg_ground = sum(r["aggregate"]["grounded_rate"] for r in comb) / n
        avg_verb = sum(r["aggregate"]["verbatim_rate"] for r in comb) / n
        avg_fill = sum(r["aggregate"]["fill_rate"] for r in comb) / n
        avg_disc = sum(r.get("discovery_time_s", 0) for r in comb) / n
        avg_ext = sum(r.get("extraction_time_s", 0) for r in comb) / n
        lines.append(f"| Combined | {avg_halluc*100:.1f}% | {avg_ground*100:.1f}% | {avg_verb*100:.1f}% | {avg_fill*100:.1f}% | {avg_disc:.1f} | {avg_ext:.1f} |")
        pipeline_aggs["combined"] = {"halluc": avg_halluc, "ground": avg_ground, "verb": avg_verb, "fill": avg_fill}

    # DSPy v2 (existing)
    if all_results.get("dspy_v2"):
        dspy = all_results["dspy_v2"]
        n = len(dspy)
        avg_halluc = sum(r["aggregate"]["hallucination_rate"] for r in dspy) / n
        avg_ground = sum(r["aggregate"]["grounded_rate"] for r in dspy) / n
        avg_verb = sum(r["aggregate"]["verbatim_rate"] for r in dspy) / n
        avg_fill = sum(r["aggregate"]["fill_rate"] for r in dspy) / n
        avg_disc = sum(r.get("discovery_time_s", 0) for r in dspy) / n
        avg_ext = sum(r.get("extraction_time_s", 0) for r in dspy) / n
        lines.append(f"| DSPy v2 | {avg_halluc*100:.1f}% | {avg_ground*100:.1f}% | {avg_verb*100:.1f}% | {avg_fill*100:.1f}% | {avg_disc:.1f} | {avg_ext:.1f} |")
        pipeline_aggs["dspy_v2"] = {"halluc": avg_halluc, "ground": avg_ground, "verb": avg_verb, "fill": avg_fill}

    # OG baselines (extract per-method averages)
    if all_results.get("baselines"):
        bl = all_results["baselines"]
        n = len(bl)
        for method in ["standard_gpt", "dspy_rlm", "rlm_lb", "rlm_v2"]:
            method_results = []
            for cluster in bl:
                m = cluster.get("methods", {}).get(method, {})
                if m and "aggregate" in m:
                    method_results.append(m["aggregate"])
            if method_results:
                mn = len(method_results)
                avg_halluc = sum(r.get("hallucination_rate", 0) for r in method_results) / mn
                avg_ground = sum(r.get("grounded_rate", 0) for r in method_results) / mn
                avg_verb = sum(r.get("verbatim_rate", 0) for r in method_results) / mn
                avg_fill = sum(r.get("fill_rate", 0) for r in method_results) / mn
                label = method.replace("_", " ").title()
                lines.append(f"| {label} (OG) | {avg_halluc*100:.1f}% | {avg_ground*100:.1f}% | {avg_verb*100:.1f}% | {avg_fill*100:.1f}% | — | — |")

    lines.append("")

    # Per-cluster breakdown
    lines.append("## Per-Cluster Breakdown")
    lines.append("")

    cluster_ids = []
    if all_results.get("rag"):
        cluster_ids = [r["cluster_id"] for r in all_results["rag"]]
    elif all_results.get("rlm_full"):
        cluster_ids = [r["cluster_id"] for r in all_results["rlm_full"]]
    elif all_results.get("combined"):
        cluster_ids = [r["cluster_id"] for r in all_results["combined"]]

    for cid in cluster_ids:
        # Find cluster label
        cluster_label = "Unknown"
        for src in ["rag", "rlm_full", "combined", "dspy_v2"]:
            for r in all_results.get(src, []):
                if r.get("cluster_id") == cid:
                    cluster_label = r["cluster_label"]
                    break
            if cluster_label != "Unknown":
                break

        lines.append(f"### {cluster_label}")
        lines.append("")
        lines.append("| Pipeline | Fields | Fill% | Verbatim% | Halluc% | Disc(s) | Ext(s) |")
        lines.append("|----------|--------|-------|-----------|---------|---------|--------|")

        for src, label in [("rag", "RAG"), ("rlm_full", "RLM Full"),
                           ("combined", "Combined"), ("dspy_v2", "DSPy v2")]:
            for r in all_results.get(src, []):
                if r.get("cluster_id") == cid:
                    a = r["aggregate"]
                    disc_t = r.get("discovery_time_s", "—")
                    ext_t = r.get("extraction_time_s", "—")
                    lines.append(
                        f"| {label} | {r.get('n_fields', '—')} | "
                        f"{a.get('fill_rate', 0)*100:.1f}% | "
                        f"{a.get('verbatim_rate', 0)*100:.1f}% | "
                        f"{a.get('hallucination_rate', 0)*100:.1f}% | "
                        f"{disc_t} | {ext_t} |"
                    )

        # OG baselines for this cluster
        for bl_cluster in all_results.get("baselines", []):
            if bl_cluster.get("cluster_id") == cid:
                for method in ["standard_gpt", "dspy_rlm"]:
                    m = bl_cluster.get("methods", {}).get(method, {})
                    if m and "aggregate" in m:
                        a = m["aggregate"]
                        label = method.replace("_", " ").title() + " (OG)"
                        lines.append(
                            f"| {label} | {m.get('n_fields', '—')} | "
                            f"{a.get('fill_rate', 0)*100:.1f}% | "
                            f"{a.get('verbatim_rate', 0)*100:.1f}% | "
                            f"{a.get('hallucination_rate', 0)*100:.1f}% | "
                            f"— | — |"
                        )

        lines.append("")

    # Analysis section
    lines.append("## Analysis")
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")

    # Determine best pipeline by hallucination rate
    if pipeline_aggs:
        best = min(pipeline_aggs.items(), key=lambda x: x[1]["halluc"])
        worst = max(pipeline_aggs.items(), key=lambda x: x[1]["halluc"])
        highest_fill = max(pipeline_aggs.items(), key=lambda x: x[1]["fill"])
        lines.append(f"- **Lowest Hallucination**: {best[0]} ({best[1]['halluc']*100:.1f}%)")
        lines.append(f"- **Highest Hallucination**: {worst[0]} ({worst[1]['halluc']*100:.1f}%)")
        lines.append(f"- **Highest Fill Rate**: {highest_fill[0]} ({highest_fill[1]['fill']*100:.1f}%)")
        lines.append("")

    lines.append("### Pipeline Architecture Comparison")
    lines.append("")
    lines.append("| Feature | Combined | RAG | RLM Full | DSPy v2 |")
    lines.append("|---------|----------|-----|----------|---------|")
    lines.append("| Retrieval | Embedding top-K | Hybrid RRF (semantic+lexical) | RLM-driven | None (all chunks) |")
    lines.append("| Context | Sequential carryforward | Per-field independent | RLM REPL state | Per-chunk independent |")
    lines.append("| Retry | Agentic (max 2) | Agentic (max 3, rephrased) | RLM iterates | None |")
    lines.append("| Compilation | Judge LLM | Reviewer only | Single-pass RLM | None |")
    lines.append("| Validation | Reviewer agent | Reviewer agent | Grounding check | Grounding check |")
    lines.append("")

    lines.append("### Trade-offs")
    lines.append("")
    lines.append("- **Combined** (Agentic): Best grounding via judge+reviewer, but slower due to sequential processing")
    lines.append("- **RAG** (Hybrid Retrieval): Good precision via targeted retrieval, but per-field LLM calls add latency")
    lines.append("- **RLM Full** (Programmatic): Most flexible — can try regex→LLM fallback, but RLM iterations are expensive")
    lines.append("- **DSPy v2** (Single-call): Fastest extraction, decent quality, but no retry mechanism")
    lines.append("")

    # Write
    md_content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(md_content)
    logger.info(f"Eval markdown written to {output_path}")
    return md_content


def main():
    db_path = "data/cuad_510_demo.duckdb"
    n_extract_chunks = 8

    # Load baseline to get cluster IDs
    baseline_path = Path("data/extraction_value_comparison_v2.json")
    with open(baseline_path) as f:
        baseline = json.load(f)
    cluster_ids = [r["cluster_id"] for r in baseline["per_cluster"]]

    logger.info(f"Running unified eval on {len(cluster_ids)} clusters")

    # Setup
    lm = _configure_dspy_lm()

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    conn = duckdb.connect(db_path, read_only=True)

    # ═══ Run new pipelines ═══
    rag_results = run_rag_pipeline(cluster_ids, conn, n_extract_chunks, embedder)
    rlm_full_results = run_rlm_full_pipeline(cluster_ids, conn, n_extract_chunks, embedder)

    conn.close()

    # ═══ Load existing results ═══
    existing = load_existing_results()

    # ═══ Assemble all results ═══
    all_results = {
        "rag": rag_results,
        "rlm_full": rlm_full_results,
        "combined": existing.get("combined", []),
        "dspy_v2": existing.get("dspy_v2", []),
        "baselines": existing.get("baselines", []),
    }

    # ═══ Save unified comparison JSON ═══
    out_path = Path("data/unified_pipeline_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nUnified results saved to {out_path}")

    # ═══ Generate eval markdown ═══
    md_path = "docs/unified_pipeline_eval.md"
    generate_eval_markdown(all_results, md_path)

    # ═══ Print summary table ═══
    print(f"\n{'='*100}")
    print(f" UNIFIED PIPELINE EVALUATION — {len(cluster_ids)} Clusters")
    print(f"{'='*100}")

    print(f"\n{'Pipeline':<16} │ {'Halluc%':>8} │ {'Ground%':>8} │ {'Verb%':>8} │ {'Fill%':>8} │ {'Time(s)':>8}")
    print(f"{'─'*16}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}")

    for label, results in [("RAG", rag_results), ("RLM Full", rlm_full_results)]:
        if results:
            n = len(results)
            h = sum(r["aggregate"]["hallucination_rate"] for r in results) / n * 100
            g = sum(r["aggregate"]["grounded_rate"] for r in results) / n * 100
            v = sum(r["aggregate"]["verbatim_rate"] for r in results) / n * 100
            f_rate = sum(r["aggregate"]["fill_rate"] for r in results) / n * 100
            t = sum(r["discovery_time_s"] + r["extraction_time_s"] for r in results) / n
            print(f"{label:<16} │ {h:>7.1f}% │ {g:>7.1f}% │ {v:>7.1f}% │ {f_rate:>7.1f}% │ {t:>7.1f}s")

    # Print existing results for comparison
    if existing.get("combined"):
        comb = existing["combined"]
        n = len(comb)
        h = sum(r["aggregate"]["hallucination_rate"] for r in comb) / n * 100
        g = sum(r["aggregate"]["grounded_rate"] for r in comb) / n * 100
        v = sum(r["aggregate"]["verbatim_rate"] for r in comb) / n * 100
        f_rate = sum(r["aggregate"]["fill_rate"] for r in comb) / n * 100
        t = sum(r.get("discovery_time_s", 0) + r.get("extraction_time_s", 0) for r in comb) / n
        print(f"{'Combined':<16} │ {h:>7.1f}% │ {g:>7.1f}% │ {v:>7.1f}% │ {f_rate:>7.1f}% │ {t:>7.1f}s")

    if existing.get("dspy_v2"):
        dspy_res = existing["dspy_v2"]
        n = len(dspy_res)
        h = sum(r["aggregate"]["hallucination_rate"] for r in dspy_res) / n * 100
        g = sum(r["aggregate"]["grounded_rate"] for r in dspy_res) / n * 100
        v = sum(r["aggregate"]["verbatim_rate"] for r in dspy_res) / n * 100
        f_rate = sum(r["aggregate"]["fill_rate"] for r in dspy_res) / n * 100
        t = sum(r.get("discovery_time_s", 0) + r.get("extraction_time_s", 0) for r in dspy_res) / n
        print(f"{'DSPy v2':<16} │ {h:>7.1f}% │ {g:>7.1f}% │ {v:>7.1f}% │ {f_rate:>7.1f}% │ {t:>7.1f}s")

    print(f"{'─'*16}─┴─{'─'*8}─┴─{'─'*8}─┴─{'─'*8}─┴─{'─'*8}─┴─{'─'*8}")
    print(f"\nFull report: {md_path}")


if __name__ == "__main__":
    main()
