#!/usr/bin/env python3
"""
3-Way Extraction Value Comparison on CLAUSE-LEVEL Clusters
==========================================================

Same comparison as extraction_value_comparison.py but runs on clause-type
clusters from the `cluster_layers` table instead of macro domain clusters.

Validates Pipeline 2 from the design doc: clause-level clustering should
produce fields that map to real contractual concepts with higher reusability.

For each clause cluster:
  1. Generate a descriptive label from sample clauses
  2. Discover fields with all 3 methods (Standard GPT, DSPy RLM, RLM+LB)
  3. Extract values from the SAME sample clauses using each method's fields
  4. Compare with grounding analysis

Usage:
    python scripts/extraction_value_comparison_clause.py data/cuad_510_demo.duckdb --n-clusters 5
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Grounding analysis ────────────────────────────────────────────────────────

def grounding_detail(value: str, chunk_text: str) -> dict:
    """Analyze how well an extracted value is grounded in the source text."""
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
    """Classify extracted value type for analysis."""
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


# ── Clause cluster label generation ──────────────────────────────────────────

def generate_cluster_label(client, deployment, clause_samples: list[str]) -> str:
    """Generate a descriptive label for a clause cluster from sample texts."""
    samples_text = "\n---\n".join(s[:500] for s in clause_samples[:5])
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{
                "role": "user",
                "content": (
                    "These are sample clauses from a single cluster of legal contract clauses. "
                    "Generate a short descriptive label (2-5 words) for this clause type.\n\n"
                    f"Sample clauses:\n{samples_text}\n\n"
                    "Respond with ONLY the label, nothing else."
                ),
            }],
            temperature=0.0,
            max_completion_tokens=30,
        )
        return resp.choices[0].message.content.strip().strip('"')
    except Exception as e:
        logger.warning(f"Label generation failed: {e}")
        return "Unknown Clause Type"


# ── Main comparison ──────────────────────────────────────────────────────────

def run_extraction_comparison_clause(db_path: str, n_clusters: int = 5,
                                      n_extract_clauses: int = 8):
    conn = duckdb.connect(db_path, read_only=True)
    config = FieldDiscoveryConfig()

    client = _get_azure_client()
    deployment = _get_deployment(config)
    logger.info(f"Azure deployment: {deployment}")

    # Get clause clusters (non-outlier) from the selected layer
    clause_clusters = conn.execute("""
        SELECT cluster_label, COUNT(*) as clause_count
        FROM cluster_layers
        WHERE clustering_run = 'clause'
          AND layer_index = 0
          AND cluster_label != -1
        GROUP BY cluster_label
        ORDER BY COUNT(*) DESC
    """).fetchdf().to_dict("records")

    logger.info(f"Found {len(clause_clusters)} clause clusters (non-outlier)")

    # Stratified sample: top (largest), mid, bottom (smallest)
    total = len(clause_clusters)
    if total <= n_clusters:
        sample = clause_clusters
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
        sample = [clause_clusters[i] for i in indices if i < total]

    logger.info(f"Sampled {len(sample)} clause clusters for comparison")
    for s in sample:
        logger.info(f"  Cluster {s['cluster_label']}: {s['clause_count']} clauses")

    METHODS = ["standard_gpt", "dspy_rlm", "rlm_lb"]
    all_results = []

    for ci, cluster in enumerate(sample):
        cluster_label_num = cluster["cluster_label"]
        clause_count = cluster["clause_count"]

        # Fetch all clause texts for this cluster
        clause_rows = conn.execute("""
            SELECT cl.item_id, c.full_text, c.clause_title, c.token_count
            FROM cluster_layers cl
            JOIN clauses c ON cl.item_id = c.clause_id
            WHERE cl.clustering_run = 'clause'
              AND cl.layer_index = 0
              AND cl.cluster_label = ?
            ORDER BY c.token_count DESC
        """, [cluster_label_num]).fetchdf().to_dict("records")

        clause_texts = [r["full_text"] for r in clause_rows if r["full_text"]]
        if len(clause_texts) < 5:
            logger.warning(f"  Cluster {cluster_label_num} has only {len(clause_texts)} clauses with text, skipping")
            continue

        # Generate a descriptive label from sample clauses
        cluster_label = generate_cluster_label(client, deployment, clause_texts[:10])
        logger.info(f"\n{'='*80}")
        logger.info(f"[{ci+1}/{len(sample)}] Clause Cluster {cluster_label_num} → \"{cluster_label}\" "
                     f"({len(clause_texts)} clauses, extracting from {min(n_extract_clauses, len(clause_texts))})")
        logger.info(f"{'='*80}")

        # Extract keywords from clause titles
        titles = [r.get("clause_title", "") for r in clause_rows if r.get("clause_title")]
        # Get most common title words as keywords
        title_words = {}
        for t in titles[:50]:
            for w in re.findall(r'\b[A-Za-z]{3,}\b', t):
                w_lower = w.lower()
                if w_lower not in {"the", "and", "for", "with", "this", "that", "from", "shall"}:
                    title_words[w_lower] = title_words.get(w_lower, 0) + 1
        keywords = [w for w, c in sorted(title_words.items(), key=lambda x: -x[1])[:10]]

        # Use the SAME sample clauses for extraction across all methods
        extract_clauses = clause_texts[:n_extract_clauses]

        cluster_result = {
            "cluster_label_num": cluster_label_num,
            "cluster_label": cluster_label,
            "clause_count": clause_count,
            "n_extract_clauses": len(extract_clauses),
            "keywords": keywords,
            "methods": {},
        }

        # ── Step 1: Discover fields with each method ──────────────────────

        # Standard GPT
        logger.info(f"  [Standard GPT] Discovering fields...")
        t0 = time.time()
        try:
            fields_std = discover_fields_for_cluster(
                client, deployment, cluster_label, keywords, clause_texts, config,
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
                cluster_label, keywords, clause_texts, config,
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
                cluster_label, keywords, clause_texts, config,
                briefing_tau=1.5, briefing_max_tokens=2000, adaptive_tau=True,
            )
        except Exception as e:
            logger.warning(f"  RLM+LB failed: {e}")
            fields_lb = []
            briefing_meta = {"error": str(e)}
        time_lb_disc = round(time.time() - t0, 2)
        logger.info(f"    → {len(fields_lb)} fields in {time_lb_disc}s")

        method_fields = {
            "standard_gpt": fields_std,
            "dspy_rlm": fields_rlm,
            "rlm_lb": fields_lb,
        }
        disc_times = {
            "standard_gpt": time_std_disc,
            "dspy_rlm": time_rlm_disc,
            "rlm_lb": time_lb_disc,
        }

        # ── Step 2: Extract values from the SAME clauses with each method ─

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

            logger.info(f"  [{method}] Extracting from {len(extract_clauses)} clauses × {len(fields)} fields...")

            t0 = time.time()
            clause_extractions = []
            for clause_idx, clause_text in enumerate(extract_clauses):
                try:
                    ext = extract_fields_from_chunk(
                        client, deployment, cluster_label, clause_text, fields, config,
                    )
                except Exception as e:
                    logger.warning(f"    Clause {clause_idx} extraction failed: {e}")
                    ext = {}

                clause_detail = {
                    "clause_index": clause_idx,
                    "clause_preview": clause_text[:200],
                    "fields": {},
                }
                for f in fields:
                    fname = f["name"]
                    if fname in ext:
                        value, confidence = ext[fname]
                        grounding = grounding_detail(value, clause_text)
                        clause_detail["fields"][fname] = {
                            "value": value,
                            "confidence": confidence,
                            "grounding": grounding,
                            "value_type": classify_value(value),
                        }
                    else:
                        clause_detail["fields"][fname] = {
                            "value": None,
                            "confidence": 0.0,
                            "grounding": {"level": "missing", "score": 0.0},
                            "value_type": "null",
                        }

                clause_extractions.append(clause_detail)

            extraction_time = round(time.time() - t0, 2)

            agg = _compute_aggregate(fields, clause_extractions)

            cluster_result["methods"][method] = {
                "n_fields": len(fields),
                "fields": fields,
                "discovery_time_s": disc_times[method],
                "extraction_time_s": extraction_time,
                "extractions": clause_extractions,
                "aggregate": agg,
                "briefing_meta": briefing_meta if method == "rlm_lb" else None,
            }

            logger.info(f"    → fill={agg['fill_rate']:.3f}, conf={agg['avg_confidence']:.3f}, "
                         f"verbatim={agg['verbatim_rate']:.3f}, halluc={agg['hallucination_rate']:.3f}, "
                         f"time={extraction_time}s")

        all_results.append(cluster_result)

    conn.close()

    # ── Generate report ──────────────────────────────────────────────────
    report = _build_report(all_results)

    # Save JSON
    out_path = Path(db_path).parent / "extraction_value_comparison_clause.json"
    with open(out_path, "w") as f:
        json.dump({"report": report, "per_cluster": all_results}, f, indent=2, default=str)
    logger.info(f"\nJSON saved to {out_path}")

    # Print report
    _print_report(report, all_results)

    # Compare with macro results if available
    macro_path = Path(db_path).parent / "extraction_value_comparison.json"
    if macro_path.exists():
        _print_macro_comparison(report, macro_path)

    return report, all_results


def _empty_aggregate():
    return {
        "fill_rate": 0, "avg_confidence": 0, "median_confidence": 0,
        "verbatim_rate": 0, "high_overlap_rate": 0, "partial_rate": 0,
        "hallucination_rate": 0, "n_extractions": 0, "n_total_slots": 0,
    }


def _compute_aggregate(fields, clause_extractions):
    """Compute aggregate extraction quality stats."""
    n_fields = len(fields)
    n_clauses = len(clause_extractions)
    total_slots = n_fields * n_clauses

    values = []
    confs = []
    groundings = []

    for ce in clause_extractions:
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
        "avg_confidence": round(float(np.mean(confs)), 3) if confs else 0,
        "median_confidence": round(float(np.median(confs)), 3) if confs else 0,
        "verbatim_rate": round(verbatim / n_extracted, 3) if n_extracted else 0,
        "high_overlap_rate": round(high_overlap / n_extracted, 3) if n_extracted else 0,
        "partial_rate": round(partial / n_extracted, 3) if n_extracted else 0,
        "hallucination_rate": round(halluc / n_extracted, 3) if n_extracted else 0,
        "grounded_rate": round((verbatim + high_overlap) / n_extracted, 3) if n_extracted else 0,
        "n_extractions": n_extracted,
        "n_total_slots": total_slots,
    }


def _build_report(all_results):
    """Build aggregate report across all clusters."""
    METHODS = ["standard_gpt", "dspy_rlm", "rlm_lb"]
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
            "avg_fields": round(float(np.mean(n_fields_list)), 1) if n_fields_list else 0,
            "failures": failures,
            "avg_fill_rate": round(float(np.mean([a["fill_rate"] for a in valid_aggs])), 3) if valid_aggs else 0,
            "avg_confidence": round(float(np.mean([a["avg_confidence"] for a in valid_aggs])), 3) if valid_aggs else 0,
            "avg_verbatim_rate": round(float(np.mean([a["verbatim_rate"] for a in valid_aggs])), 3) if valid_aggs else 0,
            "avg_grounded_rate": round(float(np.mean([a["grounded_rate"] for a in valid_aggs])), 3) if valid_aggs else 0,
            "avg_hallucination_rate": round(float(np.mean([a["hallucination_rate"] for a in valid_aggs])), 3) if valid_aggs else 0,
            "total_extractions": sum(a["n_extractions"] for a in aggs),
            "total_slots": sum(a["n_total_slots"] for a in aggs),
            "avg_discovery_time_s": round(float(np.mean(disc_times)), 1) if disc_times else 0,
            "avg_extraction_time_s": round(float(np.mean(ext_times)), 1) if ext_times else 0,
        }

    return report


def _print_report(report, all_results):
    """Print the clause-level extraction comparison report."""
    METHODS = ["standard_gpt", "dspy_rlm", "rlm_lb"]
    LABELS = {"standard_gpt": "Standard GPT", "dspy_rlm": "DSPy RLM", "rlm_lb": "RLM + LB"}
    W = 95

    print(f"\n{'='*W}")
    print(f" CLAUSE-LEVEL EXTRACTION VALUE COMPARISON")
    print(f" Standard GPT vs DSPy RLM vs RLM + Latent Briefing")
    print(f" {report['n_clusters']} clause clusters sampled")
    print(f"{'='*W}")

    rm = report["methods"]
    print(f"\n┌{'─'*40}┬{'─'*16}┬{'─'*16}┬{'─'*16}┐")
    print(f"│ {'Metric':<38} │ {'Standard GPT':>14} │ {'DSPy RLM':>14} │ {'RLM + LB':>14} │")
    print(f"├{'─'*40}┼{'─'*16}┼{'─'*16}┼{'─'*16}┤")

    rows = [
        ("FIELD DISCOVERY", "", "", ""),
        ("  Avg fields/cluster",
         f"{rm['standard_gpt']['avg_fields']:.1f}",
         f"{rm['dspy_rlm']['avg_fields']:.1f}",
         f"{rm['rlm_lb']['avg_fields']:.1f}"),
        ("  Failures (0 fields)",
         str(rm['standard_gpt']['failures']),
         str(rm['dspy_rlm']['failures']),
         str(rm['rlm_lb']['failures'])),
        ("  Avg discovery time (s)",
         f"{rm['standard_gpt']['avg_discovery_time_s']:.1f}",
         f"{rm['dspy_rlm']['avg_discovery_time_s']:.1f}",
         f"{rm['rlm_lb']['avg_discovery_time_s']:.1f}"),
        ("", "", "", ""),
        ("EXTRACTION QUALITY", "", "", ""),
        ("  Fill rate",
         f"{rm['standard_gpt']['avg_fill_rate']:.3f}",
         f"{rm['dspy_rlm']['avg_fill_rate']:.3f}",
         f"{rm['rlm_lb']['avg_fill_rate']:.3f}"),
        ("  Avg confidence",
         f"{rm['standard_gpt']['avg_confidence']:.3f}",
         f"{rm['dspy_rlm']['avg_confidence']:.3f}",
         f"{rm['rlm_lb']['avg_confidence']:.3f}"),
        ("", "", "", ""),
        ("GROUNDING ANALYSIS", "", "", ""),
        ("  Verbatim match rate",
         f"{rm['standard_gpt']['avg_verbatim_rate']:.3f}",
         f"{rm['dspy_rlm']['avg_verbatim_rate']:.3f}",
         f"{rm['rlm_lb']['avg_verbatim_rate']:.3f}"),
        ("  Grounded rate (verbatim+high)",
         f"{rm['standard_gpt']['avg_grounded_rate']:.3f}",
         f"{rm['dspy_rlm']['avg_grounded_rate']:.3f}",
         f"{rm['rlm_lb']['avg_grounded_rate']:.3f}"),
        ("  Hallucination rate",
         f"{rm['standard_gpt']['avg_hallucination_rate']:.3f}",
         f"{rm['dspy_rlm']['avg_hallucination_rate']:.3f}",
         f"{rm['rlm_lb']['avg_hallucination_rate']:.3f}"),
        ("", "", "", ""),
        ("TOTALS", "", "", ""),
        ("  Total extractions",
         str(rm['standard_gpt']['total_extractions']),
         str(rm['dspy_rlm']['total_extractions']),
         str(rm['rlm_lb']['total_extractions'])),
        ("  Avg extraction time (s)",
         f"{rm['standard_gpt']['avg_extraction_time_s']:.1f}",
         f"{rm['dspy_rlm']['avg_extraction_time_s']:.1f}",
         f"{rm['rlm_lb']['avg_extraction_time_s']:.1f}"),
    ]

    for label, v1, v2, v3 in rows:
        if not label:
            print(f"├{'─'*40}┼{'─'*16}┼{'─'*16}┼{'─'*16}┤")
        else:
            print(f"│ {label:<38} │ {v1:>14} │ {v2:>14} │ {v3:>14} │")

    print(f"└{'─'*40}┴{'─'*16}┴{'─'*16}┴{'─'*16}┘")

    # ── Per-cluster summary ──
    print(f"\n{'─'*W}")
    print(f" Per-Cluster Extraction Summary (Clause-Level)")
    print(f"{'─'*W}")
    header = (f"{'Cluster':<30} │ {'Std':>4} {'RLM':>4} {'LB':>4} │ "
              f"{'S.Fill':>6} {'R.Fill':>6} {'L.Fill':>6} │ "
              f"{'S.Vrb':>5} {'R.Vrb':>5} {'L.Vrb':>5} │ "
              f"{'S.Hal':>5} {'R.Hal':>5} {'L.Hal':>5}")
    print(header)
    print(f"{'─'*W}")

    for r in all_results:
        label = f"[{r['cluster_label_num']}] {r['cluster_label']}"[:28]
        s = r["methods"].get("standard_gpt", {})
        d = r["methods"].get("dspy_rlm", {})
        l = r["methods"].get("rlm_lb", {})

        sa = s.get("aggregate", _empty_aggregate())
        da = d.get("aggregate", _empty_aggregate())
        la = l.get("aggregate", _empty_aggregate())

        print(f"{label:<30} │ {s.get('n_fields',0):>4} {d.get('n_fields',0):>4} {l.get('n_fields',0):>4} │ "
              f"{sa['fill_rate']:>6.3f} {da['fill_rate']:>6.3f} {la['fill_rate']:>6.3f} │ "
              f"{sa['verbatim_rate']:>5.3f} {da['verbatim_rate']:>5.3f} {la['verbatim_rate']:>5.3f} │ "
              f"{sa['hallucination_rate']:>5.3f} {da['hallucination_rate']:>5.3f} {la['hallucination_rate']:>5.3f}")

    # ── Value-level extraction samples ──
    print(f"\n{'─'*W}")
    print(f" Extraction Value Samples (first 2 clauses per cluster)")
    print(f"{'─'*W}")

    for r in all_results:
        label = f"[{r['cluster_label_num']}] {r['cluster_label']}"
        print(f"\n  ┌─ {label}")

        all_field_names = set()
        for method in METHODS:
            m = r["methods"].get(method, {})
            for f in m.get("fields", []):
                all_field_names.add(f["name"])

        for clause_idx in range(min(2, r.get("n_extract_clauses", 0))):
            print(f"  │")
            print(f"  │  Clause {clause_idx + 1}:")

            method_vals = {}
            for method in METHODS:
                m = r["methods"].get(method, {})
                exts = m.get("extractions", [])
                if clause_idx < len(exts):
                    method_vals[method] = exts[clause_idx]["fields"]
                else:
                    method_vals[method] = {}

            active_fields = set()
            for method in METHODS:
                for fname, fdata in method_vals[method].items():
                    if fdata.get("value") is not None:
                        active_fields.add(fname)

            if not active_fields:
                print(f"  │    (no extractions)")
                continue

            print(f"  │    {'Field':<25} {'Standard GPT':<30} {'DSPy RLM':<30} {'RLM+LB':<30}")
            print(f"  │    {'─'*25} {'─'*30} {'─'*30} {'─'*30}")

            for fname in sorted(active_fields):
                vals = []
                for method in METHODS:
                    fdata = method_vals[method].get(fname, {})
                    val = fdata.get("value")
                    if val is None:
                        vals.append("—")
                    else:
                        grnd = fdata.get("grounding", {}).get("level", "?")
                        short_val = val[:22] if len(val) > 22 else val
                        tag = {"verbatim": "✓", "high_overlap": "~", "partial_overlap": "◦",
                               "weak_overlap": "?", "hallucinated": "✗"}.get(grnd, "?")
                        vals.append(f"{short_val} [{tag}]")

                print(f"  │    {fname:<25} {vals[0]:<30} {vals[1]:<30} {vals[2]:<30}")

        print(f"  └─")

    # ── Grounding distribution ──
    print(f"\n{'─'*W}")
    print(f" Grounding Distribution (all extractions)")
    print(f"{'─'*W}")
    print(f"  {'Level':<20} {'Standard GPT':>14} {'DSPy RLM':>14} {'RLM + LB':>14}")
    print(f"  {'─'*62}")

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

        print(f"  {level:<20} {counts[0]:>14} {counts[1]:>14} {counts[2]:>14}")

    # ── Winner determination ──
    print(f"\n{'='*W}")
    print(f" WINNER ANALYSIS (Clause-Level)")
    print(f"{'='*W}")

    best_fill = max(METHODS, key=lambda m: rm[m]["avg_fill_rate"])
    best_conf = max(METHODS, key=lambda m: rm[m]["avg_confidence"])
    best_grounded = max(METHODS, key=lambda m: rm[m]["avg_grounded_rate"])
    best_verbatim = max(METHODS, key=lambda m: rm[m]["avg_verbatim_rate"])
    least_halluc = min(METHODS, key=lambda m: rm[m]["avg_hallucination_rate"])
    most_fields = max(METHODS, key=lambda m: rm[m]["avg_fields"])

    print(f"  Most fields discovered:     {LABELS[most_fields]} ({rm[most_fields]['avg_fields']:.1f} avg)")
    print(f"  Best fill rate:             {LABELS[best_fill]} ({rm[best_fill]['avg_fill_rate']:.3f})")
    print(f"  Highest confidence:         {LABELS[best_conf]} ({rm[best_conf]['avg_confidence']:.3f})")
    print(f"  Best verbatim match:        {LABELS[best_verbatim]} ({rm[best_verbatim]['avg_verbatim_rate']:.3f})")
    print(f"  Best grounded rate:         {LABELS[best_grounded]} ({rm[best_grounded]['avg_grounded_rate']:.3f})")
    print(f"  Lowest hallucination:       {LABELS[least_halluc]} ({rm[least_halluc]['avg_hallucination_rate']:.3f})")
    print(f"{'='*W}\n")


def _print_macro_comparison(clause_report, macro_path):
    """Print side-by-side comparison of clause vs macro cluster results."""
    with open(macro_path) as f:
        macro_data = json.load(f)
    macro_report = macro_data["report"]

    METHODS = ["standard_gpt", "dspy_rlm", "rlm_lb"]
    LABELS = {"standard_gpt": "Standard GPT", "dspy_rlm": "DSPy RLM", "rlm_lb": "RLM + LB"}
    W = 95

    print(f"\n{'='*W}")
    print(f" CLAUSE vs MACRO CLUSTER COMPARISON")
    print(f" Clause: {clause_report['n_clusters']} clusters | Macro: {macro_report['n_clusters']} clusters")
    print(f"{'='*W}")

    for method in METHODS:
        cm = clause_report["methods"][method]
        mm = macro_report["methods"][method]

        print(f"\n  {LABELS[method]}:")
        print(f"  {'Metric':<30} {'Macro':>10} {'Clause':>10} {'Delta':>10}")
        print(f"  {'─'*60}")

        metrics = [
            ("Avg fields/cluster", "avg_fields", ".1f"),
            ("Fill rate", "avg_fill_rate", ".3f"),
            ("Avg confidence", "avg_confidence", ".3f"),
            ("Verbatim match rate", "avg_verbatim_rate", ".3f"),
            ("Grounded rate", "avg_grounded_rate", ".3f"),
            ("Hallucination rate", "avg_hallucination_rate", ".3f"),
            ("Avg discovery time (s)", "avg_discovery_time_s", ".1f"),
            ("Avg extraction time (s)", "avg_extraction_time_s", ".1f"),
        ]

        for label, key, fmt in metrics:
            mv = mm.get(key, 0)
            cv = cm.get(key, 0)
            delta = cv - mv
            sign = "+" if delta > 0 else ""
            print(f"  {label:<30} {mv:>10{fmt}} {cv:>10{fmt}} {sign}{delta:>9{fmt}}")

    print(f"\n{'='*W}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-way extraction value comparison on clause clusters")
    parser.add_argument("db_path", help="Path to DuckDB file")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clause clusters (default: 5)")
    parser.add_argument("--n-clauses", type=int, default=8, help="Clauses to extract per cluster (default: 8)")
    args = parser.parse_args()

    run_extraction_comparison_clause(args.db_path, n_clusters=args.n_clusters,
                                      n_extract_clauses=args.n_clauses)
