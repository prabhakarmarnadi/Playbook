#!/usr/bin/env python3
"""
4-Way Extraction Value Comparison: Standard GPT vs DSPy RLM vs RLM+LB vs RLM v2
=================================================================================

Extends the 3-way comparison with the spec-aligned RLM v2 pipeline that adds:
  - Rolling Structured State across chunk batches
  - Dual-mode exploration (grounded + exploratory)
  - Four output types (fields, clause_types, intents, obligations)
  - Overlap-penalty merge + two-tier canonicalization

Usage:
    python scripts/extraction_value_comparison_v2.py data/cuad_510_demo.duckdb --n-clusters 5
"""
import argparse
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

from core.field_discovery import (
    FieldDiscoveryConfig,
    discover_fields_for_cluster,
    discover_fields_for_cluster_rlm,
    discover_fields_for_cluster_rlm_lb,
    extract_fields_from_chunk,
    _get_azure_client,
    _get_deployment,
)
from core.rlm_v2 import discover_fields_for_cluster_rlm_v2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Grounding analysis (same as v1) ──────────────────────────────────────────

def grounding_detail(value: str, chunk_text: str) -> dict:
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
    token_ratio = len(overlap) / len(v_tokens)
    if token_ratio >= 0.8:
        return {"level": "high_overlap", "score": 0.8, "token_ratio": round(token_ratio, 3)}
    elif token_ratio >= 0.5:
        return {"level": "partial_overlap", "score": 0.5, "token_ratio": round(token_ratio, 3)}
    elif token_ratio >= 0.2:
        return {"level": "weak_overlap", "score": 0.2, "token_ratio": round(token_ratio, 3)}
    else:
        return {"level": "hallucinated", "score": 0.0, "token_ratio": round(token_ratio, 3)}


def classify_value(value: str) -> str:
    if not value:
        return "null"
    v = value.strip()
    if re.match(r'^\d+$', v):
        return "integer"
    if re.match(r'^\$?[\d,]+\.?\d*$', v):
        return "currency"
    if re.match(r'^\d+(\.\d+)?%$', v):
        return "percentage"
    if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', v):
        return "date"
    if v.lower() in ("true", "false", "yes", "no"):
        return "boolean"
    if len(v) > 100:
        return "long_text"
    return "short_text"


# ── Main comparison ──────────────────────────────────────────────────────────

def run_extraction_comparison(db_path: str, n_clusters: int = 5,
                              n_extract_chunks: int = 8):
    conn = duckdb.connect(db_path, read_only=True)
    config = FieldDiscoveryConfig()

    client = _get_azure_client()
    deployment = _get_deployment(config)
    logger.info(f"Azure deployment: {deployment}")

    # Get clusters — stratified sample (same as v1)
    all_clusters = conn.execute("""
        SELECT c.cluster_id, c.label, c.keywords, c.chunk_count
        FROM clusters c
        WHERE c.chunk_count >= 15
        ORDER BY c.chunk_count DESC
    """).fetchdf().to_dict("records")

    logger.info(f"Found {len(all_clusters)} clusters with ≥15 chunks")

    total = len(all_clusters)
    if total <= n_clusters:
        sample = all_clusters
    else:
        n_top = max(1, int(n_clusters * 0.4))
        n_mid = max(1, int(n_clusters * 0.3))
        n_bot = n_clusters - n_top - n_mid
        top_idx = list(range(n_top))
        mid_start = total // 3
        mid_idx = list(range(mid_start, mid_start + n_mid))
        bot_start = max(total - n_bot, mid_start + n_mid)
        bot_idx = list(range(bot_start, bot_start + n_bot))
        indices = sorted(set(top_idx + mid_idx + bot_idx))
        sample = [all_clusters[i] for i in indices if i < total]

    logger.info(f"Sampled {len(sample)} clusters for 4-way comparison")

    # Pre-load embedder for v2 (reuse across clusters)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Loaded MiniLM-L6-v2 for v2 embedding scoring")

    METHODS = ["standard_gpt", "dspy_rlm", "rlm_lb", "rlm_v2"]
    all_results = []

    for ci, cluster in enumerate(sample):
        cluster_id = cluster["cluster_id"]
        cluster_label = cluster["label"] or "Unknown"
        chunk_count = cluster["chunk_count"]

        try:
            keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(cluster.get("keywords"), str) else (cluster.get("keywords") or [])
        except json.JSONDecodeError:
            keywords = []

        chunk_rows = conn.execute("""
            SELECT ch.chunk_id, ch.chunk_text
            FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ca.cluster_id = ?
            ORDER BY ch.chunk_index
        """, [cluster_id]).fetchdf().to_dict("records")

        chunk_texts = [r["chunk_text"] for r in chunk_rows]
        if len(chunk_texts) < 5:
            continue

        extract_chunks = chunk_texts[:n_extract_chunks]

        logger.info(f"\n{'='*80}")
        logger.info(f"[{ci+1}/{len(sample)}] {cluster_label} ({len(chunk_texts)} chunks, extracting from {len(extract_chunks)})")
        logger.info(f"{'='*80}")

        cluster_result = {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "chunk_count": chunk_count,
            "n_extract_chunks": len(extract_chunks),
            "methods": {},
        }

        # ── Step 1: Discover fields with each method ──────────────────────

        # Standard GPT
        logger.info(f"  [Standard GPT] Discovering fields...")
        t0 = time.time()
        try:
            fields_std = discover_fields_for_cluster(
                client, deployment, cluster_label, keywords, chunk_texts, config,
            )
        except Exception as e:
            logger.warning(f"  Standard GPT failed: {e}")
            fields_std = []
        time_std_disc = round(time.time() - t0, 2)
        logger.info(f"    → {len(fields_std)} fields in {time_std_disc}s")

        # DSPy RLM
        logger.info(f"  [DSPy RLM] Discovering fields...")
        t0 = time.time()
        try:
            fields_rlm = discover_fields_for_cluster_rlm(
                cluster_label, keywords, chunk_texts, config,
            )
        except Exception as e:
            logger.warning(f"  DSPy RLM failed: {e}")
            fields_rlm = []
        time_rlm_disc = round(time.time() - t0, 2)
        logger.info(f"    → {len(fields_rlm)} fields in {time_rlm_disc}s")

        # RLM + Latent Briefing
        logger.info(f"  [RLM+LB] Discovering fields...")
        t0 = time.time()
        try:
            fields_lb, briefing_meta = discover_fields_for_cluster_rlm_lb(
                cluster_label, keywords, chunk_texts, config,
                briefing_tau=1.5, briefing_max_tokens=2000, adaptive_tau=True,
            )
        except Exception as e:
            logger.warning(f"  RLM+LB failed: {e}")
            fields_lb = []
            briefing_meta = {"error": str(e)}
        time_lb_disc = round(time.time() - t0, 2)
        logger.info(f"    → {len(fields_lb)} fields in {time_lb_disc}s")

        # RLM v2 (spec-aligned)
        logger.info(f"  [RLM v2] Discovering fields (rolling state + dual mode)...")
        t0 = time.time()
        try:
            fields_v2, v2_meta = discover_fields_for_cluster_rlm_v2(
                client, deployment, cluster_label, keywords, chunk_texts, config,
                embedder=embedder,
            )
        except Exception as e:
            logger.warning(f"  RLM v2 failed: {e}")
            fields_v2 = []
            v2_meta = {"error": str(e)}
        time_v2_disc = round(time.time() - t0, 2)
        logger.info(f"    → {len(fields_v2)} fields in {time_v2_disc}s")
        if v2_meta.get("n_clause_types"):
            logger.info(f"      + {v2_meta['n_clause_types']} clause types, "
                         f"{v2_meta.get('n_intents', 0)} intents, "
                         f"{v2_meta.get('n_obligations', 0)} obligations")

        method_fields = {
            "standard_gpt": fields_std,
            "dspy_rlm": fields_rlm,
            "rlm_lb": fields_lb,
            "rlm_v2": fields_v2,
        }
        disc_times = {
            "standard_gpt": time_std_disc,
            "dspy_rlm": time_rlm_disc,
            "rlm_lb": time_lb_disc,
            "rlm_v2": time_v2_disc,
        }

        # ── Step 2: Extract values from the SAME chunks with each method ──

        for method in METHODS:
            fields = method_fields[method]
            if not fields:
                cluster_result["methods"][method] = {
                    "n_fields": 0,
                    "fields": [],
                    "discovery_time_s": disc_times[method],
                    "extraction_time_s": 0,
                    "extractions": [],
                    "aggregate": _empty_aggregate(),
                }
                continue

            logger.info(f"  [{method}] Extracting from {len(extract_chunks)} chunks × {len(fields)} fields...")

            # Build RLM context for v2 method
            rlm_context = None
            if method == "rlm_v2" and v2_meta:
                rlm_context = {
                    "clause_types": v2_meta.get("clause_types", []),
                    "intents": v2_meta.get("intents", []),
                    "obligations": v2_meta.get("obligations", []),
                }

            t0 = time.time()
            chunk_extractions = []
            for chunk_idx, chunk_text in enumerate(extract_chunks):
                try:
                    ext = extract_fields_from_chunk(
                        client, deployment, cluster_label, chunk_text, fields, config,
                        rlm_context=rlm_context,
                    )
                except Exception as e:
                    logger.warning(f"    Chunk {chunk_idx} extraction failed: {e}")
                    ext = {}

                chunk_detail = {
                    "chunk_index": chunk_idx,
                    "chunk_preview": chunk_text[:200],
                    "fields": {},
                }
                for f in fields:
                    fname = f["name"]
                    if fname in ext:
                        value, confidence = ext[fname]
                        grounding = grounding_detail(value, chunk_text)
                        chunk_detail["fields"][fname] = {
                            "value": value,
                            "confidence": confidence,
                            "grounding": grounding,
                            "value_type": classify_value(value),
                        }
                    else:
                        chunk_detail["fields"][fname] = {
                            "value": None,
                            "confidence": 0.0,
                            "grounding": {"level": "missing", "score": 0.0},
                            "value_type": "null",
                        }

                chunk_extractions.append(chunk_detail)

            extraction_time = round(time.time() - t0, 2)
            agg = _compute_aggregate(fields, chunk_extractions)

            extra_meta = {}
            if method == "rlm_lb":
                extra_meta["briefing_meta"] = briefing_meta
            elif method == "rlm_v2":
                extra_meta["v2_meta"] = v2_meta

            cluster_result["methods"][method] = {
                "n_fields": len(fields),
                "fields": fields,
                "discovery_time_s": disc_times[method],
                "extraction_time_s": extraction_time,
                "extractions": chunk_extractions,
                "aggregate": agg,
                **extra_meta,
            }

            logger.info(f"    → fill={agg['fill_rate']:.3f}, conf={agg['avg_confidence']:.3f}, "
                         f"verbatim={agg['verbatim_rate']:.3f}, halluc={agg['hallucination_rate']:.3f}, "
                         f"time={extraction_time}s")

        all_results.append(cluster_result)

    conn.close()

    # ── Generate report ──────────────────────────────────────────────────
    report = _build_report(all_results)

    out_path = Path(db_path).parent / "extraction_value_comparison_v2.json"
    with open(out_path, "w") as f:
        json.dump({"report": report, "per_cluster": all_results}, f, indent=2, default=str)
    logger.info(f"\nJSON saved to {out_path}")

    _print_report(report, all_results)

    return report, all_results


def _empty_aggregate():
    return {
        "fill_rate": 0, "avg_confidence": 0, "median_confidence": 0,
        "verbatim_rate": 0, "high_overlap_rate": 0, "partial_rate": 0,
        "hallucination_rate": 0, "n_extractions": 0, "n_total_slots": 0,
    }


def _compute_aggregate(fields, chunk_extractions):
    n_fields = len(fields)
    n_chunks = len(chunk_extractions)
    total_slots = n_fields * n_chunks

    values, confs, groundings = [], [], []

    for ce in chunk_extractions:
        for f in fields:
            fname = f["name"]
            fd = ce["fields"].get(fname, {})
            val = fd.get("value")
            if val is not None:
                values.append(val)
                confs.append(fd.get("confidence", 0))
                groundings.append(fd.get("grounding", {}).get("level", "missing"))

    n_extracted = len(values)
    fill_rate = n_extracted / total_slots if total_slots else 0

    verbatim = sum(1 for g in groundings if g == "verbatim")
    high_overlap = sum(1 for g in groundings if g == "high_overlap")
    partial = sum(1 for g in groundings if g in ("partial_overlap", "weak_overlap"))
    halluc = sum(1 for g in groundings if g == "hallucinated")

    return {
        "fill_rate": round(fill_rate, 3),
        "avg_confidence": round(np.mean(confs), 3) if confs else 0,
        "median_confidence": round(np.median(confs), 3) if confs else 0,
        "verbatim_rate": round(verbatim / n_extracted, 3) if n_extracted else 0,
        "high_overlap_rate": round(high_overlap / n_extracted, 3) if n_extracted else 0,
        "partial_rate": round(partial / n_extracted, 3) if n_extracted else 0,
        "hallucination_rate": round(halluc / n_extracted, 3) if n_extracted else 0,
        "grounded_rate": round((verbatim + high_overlap) / n_extracted, 3) if n_extracted else 0,
        "n_extractions": n_extracted,
        "n_total_slots": total_slots,
    }


def _build_report(all_results):
    METHODS = ["standard_gpt", "dspy_rlm", "rlm_lb", "rlm_v2"]
    report = {"n_clusters": len(all_results), "methods": {}}

    for method in METHODS:
        aggs = [r["methods"][method]["aggregate"] for r in all_results
                if r["methods"].get(method)]
        n_fields_list = [r["methods"][method]["n_fields"] for r in all_results
                         if r["methods"].get(method)]
        disc_times = [r["methods"][method]["discovery_time_s"] for r in all_results
                      if r["methods"].get(method)]
        ext_times = [r["methods"][method]["extraction_time_s"] for r in all_results
                     if r["methods"].get(method)]

        failures = sum(1 for nf in n_fields_list if nf == 0)
        valid_aggs = [a for a in aggs if a["n_extractions"] > 0]

        report["methods"][method] = {
            "avg_fields": round(np.mean(n_fields_list), 1) if n_fields_list else 0,
            "failures": failures,
            "avg_fill_rate": round(np.mean([a["fill_rate"] for a in valid_aggs]), 3) if valid_aggs else 0,
            "avg_confidence": round(np.mean([a["avg_confidence"] for a in valid_aggs]), 3) if valid_aggs else 0,
            "avg_verbatim_rate": round(np.mean([a["verbatim_rate"] for a in valid_aggs]), 3) if valid_aggs else 0,
            "avg_grounded_rate": round(np.mean([a["grounded_rate"] for a in valid_aggs]), 3) if valid_aggs else 0,
            "avg_hallucination_rate": round(np.mean([a["hallucination_rate"] for a in valid_aggs]), 3) if valid_aggs else 0,
            "total_extractions": sum(a["n_extractions"] for a in aggs),
            "total_slots": sum(a["n_total_slots"] for a in aggs),
            "avg_discovery_time_s": round(np.mean(disc_times), 1) if disc_times else 0,
            "avg_extraction_time_s": round(np.mean(ext_times), 1) if ext_times else 0,
        }

    return report


def _print_report(report, all_results):
    METHODS = ["standard_gpt", "dspy_rlm", "rlm_lb", "rlm_v2"]
    LABELS = {"standard_gpt": "Standard GPT", "dspy_rlm": "DSPy RLM",
              "rlm_lb": "RLM + LB", "rlm_v2": "RLM v2"}
    W = 115

    print(f"\n{'='*W}")
    print(f" 4-WAY EXTRACTION VALUE COMPARISON")
    print(f" Standard GPT vs DSPy RLM vs RLM + Latent Briefing vs RLM v2 (Spec-Aligned)")
    print(f" {report['n_clusters']} clusters, {8} chunks/cluster extraction sample")
    print(f"{'='*W}")

    rm = report["methods"]
    print(f"\n┌{'─'*40}┬{'─'*16}┬{'─'*16}┬{'─'*16}┬{'─'*16}┐")
    print(f"│ {'Metric':<38} │ {'Standard GPT':>14} │ {'DSPy RLM':>14} │ {'RLM + LB':>14} │ {'RLM v2':>14} │")
    print(f"├{'─'*40}┼{'─'*16}┼{'─'*16}┼{'─'*16}┼{'─'*16}┤")

    def _v(method, key, fmt=".1f"):
        val = rm.get(method, {}).get(key, 0)
        return f"{val:{fmt}}"

    rows = [
        ("FIELD DISCOVERY", "", "", "", ""),
        ("  Avg fields/cluster",
         _v("standard_gpt", "avg_fields"),
         _v("dspy_rlm", "avg_fields"),
         _v("rlm_lb", "avg_fields"),
         _v("rlm_v2", "avg_fields")),
        ("  Failures (0 fields)",
         str(rm.get("standard_gpt", {}).get("failures", 0)),
         str(rm.get("dspy_rlm", {}).get("failures", 0)),
         str(rm.get("rlm_lb", {}).get("failures", 0)),
         str(rm.get("rlm_v2", {}).get("failures", 0))),
        ("  Avg discovery time (s)",
         _v("standard_gpt", "avg_discovery_time_s"),
         _v("dspy_rlm", "avg_discovery_time_s"),
         _v("rlm_lb", "avg_discovery_time_s"),
         _v("rlm_v2", "avg_discovery_time_s")),
        ("", "", "", "", ""),
        ("EXTRACTION QUALITY", "", "", "", ""),
        ("  Fill rate",
         _v("standard_gpt", "avg_fill_rate", ".3f"),
         _v("dspy_rlm", "avg_fill_rate", ".3f"),
         _v("rlm_lb", "avg_fill_rate", ".3f"),
         _v("rlm_v2", "avg_fill_rate", ".3f")),
        ("  Avg confidence",
         _v("standard_gpt", "avg_confidence", ".3f"),
         _v("dspy_rlm", "avg_confidence", ".3f"),
         _v("rlm_lb", "avg_confidence", ".3f"),
         _v("rlm_v2", "avg_confidence", ".3f")),
        ("", "", "", "", ""),
        ("GROUNDING ANALYSIS", "", "", "", ""),
        ("  Verbatim match rate",
         _v("standard_gpt", "avg_verbatim_rate", ".3f"),
         _v("dspy_rlm", "avg_verbatim_rate", ".3f"),
         _v("rlm_lb", "avg_verbatim_rate", ".3f"),
         _v("rlm_v2", "avg_verbatim_rate", ".3f")),
        ("  Grounded rate (verbatim+high)",
         _v("standard_gpt", "avg_grounded_rate", ".3f"),
         _v("dspy_rlm", "avg_grounded_rate", ".3f"),
         _v("rlm_lb", "avg_grounded_rate", ".3f"),
         _v("rlm_v2", "avg_grounded_rate", ".3f")),
        ("  Hallucination rate",
         _v("standard_gpt", "avg_hallucination_rate", ".3f"),
         _v("dspy_rlm", "avg_hallucination_rate", ".3f"),
         _v("rlm_lb", "avg_hallucination_rate", ".3f"),
         _v("rlm_v2", "avg_hallucination_rate", ".3f")),
        ("", "", "", "", ""),
        ("TOTALS", "", "", "", ""),
        ("  Total extractions",
         str(rm.get("standard_gpt", {}).get("total_extractions", 0)),
         str(rm.get("dspy_rlm", {}).get("total_extractions", 0)),
         str(rm.get("rlm_lb", {}).get("total_extractions", 0)),
         str(rm.get("rlm_v2", {}).get("total_extractions", 0))),
        ("  Avg extraction time (s)",
         _v("standard_gpt", "avg_extraction_time_s"),
         _v("dspy_rlm", "avg_extraction_time_s"),
         _v("rlm_lb", "avg_extraction_time_s"),
         _v("rlm_v2", "avg_extraction_time_s")),
    ]

    for label, v1, v2, v3, v4 in rows:
        if not label:
            print(f"├{'─'*40}┼{'─'*16}┼{'─'*16}┼{'─'*16}┼{'─'*16}┤")
        else:
            print(f"│ {label:<38} │ {v1:>14} │ {v2:>14} │ {v3:>14} │ {v4:>14} │")

    print(f"└{'─'*40}┴{'─'*16}┴{'─'*16}┴{'─'*16}┴{'─'*16}┘")

    # ── Per-cluster summary ──
    print(f"\n{'─'*W}")
    print(f" Per-Cluster Extraction Summary")
    print(f"{'─'*W}")
    hdr = (f"{'Cluster':<25} │ "
           f"{'S':>3} {'R':>3} {'L':>3} {'V':>3} │ "
           f"{'S.Fil':>5} {'R.Fil':>5} {'L.Fil':>5} {'V.Fil':>5} │ "
           f"{'S.Vrb':>5} {'R.Vrb':>5} {'L.Vrb':>5} {'V.Vrb':>5} │ "
           f"{'S.Hal':>5} {'R.Hal':>5} {'L.Hal':>5} {'V.Hal':>5}")
    print(hdr)
    print(f"{'─'*W}")

    for r in all_results:
        label = r["cluster_label"][:23]
        s = r["methods"].get("standard_gpt", {})
        d = r["methods"].get("dspy_rlm", {})
        l = r["methods"].get("rlm_lb", {})
        v = r["methods"].get("rlm_v2", {})

        sa = s.get("aggregate", _empty_aggregate())
        da = d.get("aggregate", _empty_aggregate())
        la = l.get("aggregate", _empty_aggregate())
        va = v.get("aggregate", _empty_aggregate())

        print(f"{label:<25} │ "
              f"{s.get('n_fields',0):>3} {d.get('n_fields',0):>3} {l.get('n_fields',0):>3} {v.get('n_fields',0):>3} │ "
              f"{sa['fill_rate']:>5.3f} {da['fill_rate']:>5.3f} {la['fill_rate']:>5.3f} {va['fill_rate']:>5.3f} │ "
              f"{sa['verbatim_rate']:>5.3f} {da['verbatim_rate']:>5.3f} {la['verbatim_rate']:>5.3f} {va['verbatim_rate']:>5.3f} │ "
              f"{sa['hallucination_rate']:>5.3f} {da['hallucination_rate']:>5.3f} {la['hallucination_rate']:>5.3f} {va['hallucination_rate']:>5.3f}")

    # ── RLM v2 bonus outputs ──
    print(f"\n{'─'*W}")
    print(f" RLM v2 Bonus Outputs (beyond fields)")
    print(f"{'─'*W}")
    for r in all_results:
        v2m = r["methods"].get("rlm_v2", {})
        v2_meta = v2m.get("v2_meta", {})
        print(f"  {r['cluster_label'][:30]:<32} "
              f"clause_types={v2_meta.get('n_clause_types', 0):>2}, "
              f"intents={v2_meta.get('n_intents', 0):>2}, "
              f"obligations={v2_meta.get('n_obligations', 0):>2}, "
              f"batches={v2_meta.get('iterations', 0)}, "
              f"explore_ratio={v2_meta.get('exploration_ratio', 0):.2f}, "
              f"canon_removed={v2_meta.get('canonicalization_removed', 0)}")

    # ── Grounding distribution ──
    print(f"\n{'─'*W}")
    print(f" Grounding Distribution (all extractions)")
    print(f"{'─'*W}")
    print(f"  {'Level':<20} {'Standard GPT':>14} {'DSPy RLM':>14} {'RLM + LB':>14} {'RLM v2':>14}")
    print(f"  {'─'*78}")

    for level in ["verbatim", "high_overlap", "partial_overlap", "weak_overlap", "hallucinated"]:
        counts = []
        for method in METHODS:
            total = 0
            for r in all_results:
                m = r["methods"].get(method, {})
                for ext in m.get("extractions", []):
                    for fname, fdata in ext.get("fields", {}).items():
                        if fdata.get("value") is not None and fdata.get("grounding", {}).get("level") == level:
                            total += 1
            counts.append(total)
        print(f"  {level:<20} {counts[0]:>14} {counts[1]:>14} {counts[2]:>14} {counts[3]:>14}")

    # ── Winner determination ──
    print(f"\n{'='*W}")
    print(f" WINNER ANALYSIS")
    print(f"{'='*W}")

    best_fill = max(METHODS, key=lambda m: rm.get(m, {}).get("avg_fill_rate", 0))
    best_conf = max(METHODS, key=lambda m: rm.get(m, {}).get("avg_confidence", 0))
    best_grounded = max(METHODS, key=lambda m: rm.get(m, {}).get("avg_grounded_rate", 0))
    best_verbatim = max(METHODS, key=lambda m: rm.get(m, {}).get("avg_verbatim_rate", 0))
    least_halluc = min(METHODS, key=lambda m: rm.get(m, {}).get("avg_hallucination_rate", 1))
    most_fields = max(METHODS, key=lambda m: rm.get(m, {}).get("avg_fields", 0))

    print(f"  Most fields discovered:     {LABELS[most_fields]} ({rm.get(most_fields, {}).get('avg_fields', 0):.1f} avg)")
    print(f"  Best fill rate:             {LABELS[best_fill]} ({rm.get(best_fill, {}).get('avg_fill_rate', 0):.3f})")
    print(f"  Highest confidence:         {LABELS[best_conf]} ({rm.get(best_conf, {}).get('avg_confidence', 0):.3f})")
    print(f"  Best verbatim match:        {LABELS[best_verbatim]} ({rm.get(best_verbatim, {}).get('avg_verbatim_rate', 0):.3f})")
    print(f"  Best grounded rate:         {LABELS[best_grounded]} ({rm.get(best_grounded, {}).get('avg_grounded_rate', 0):.3f})")
    print(f"  Lowest hallucination:       {LABELS[least_halluc]} ({rm.get(least_halluc, {}).get('avg_hallucination_rate', 0):.3f})")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-way extraction value comparison (with RLM v2)")
    parser.add_argument("db_path", help="Path to DuckDB file")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters (default: 5)")
    parser.add_argument("--n-chunks", type=int, default=8, help="Chunks to extract per cluster (default: 8)")
    args = parser.parse_args()

    run_extraction_comparison(args.db_path, n_clusters=args.n_clusters,
                              n_extract_chunks=args.n_chunks)
