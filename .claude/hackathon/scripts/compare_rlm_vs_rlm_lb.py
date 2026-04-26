#!/usr/bin/env python3
"""
Compare Vanilla RLM vs RLM + Latent Briefing on field discovery + extraction.

Runs both approaches on the same clusters and compares:
  - Number of fields discovered
  - Field overlap (Jaccard, near-matches)
  - Extraction fill rate and confidence (grounded)
  - Latency
  - Latent Briefing compression stats

Usage:
    python scripts/compare_rlm_vs_rlm_lb.py data/cuad_510_demo.duckdb --n-clusters 5
    python scripts/compare_rlm_vs_rlm_lb.py data/cuad_510_demo.duckdb --n-clusters 10 --skip-extraction
"""
import argparse
import json
import logging
import os
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


# ── Metrics ───────────────────────────────────────────────────────────────────

def field_overlap(fields_a: list[dict], fields_b: list[dict]) -> dict:
    """Compute overlap metrics between two field lists."""
    names_a = {f["name"] for f in fields_a}
    names_b = {f["name"] for f in fields_b}

    intersection = names_a & names_b
    union = names_a | names_b

    jaccard = len(intersection) / len(union) if union else 1.0

    # Near-match: substring containment
    near_matches_a_to_b = 0
    for na in names_a - names_b:
        for nb in names_b - names_a:
            if na in nb or nb in na:
                near_matches_a_to_b += 1
                break

    return {
        "count_a": len(names_a),
        "count_b": len(names_b),
        "exact_overlap": len(intersection),
        "jaccard": round(jaccard, 3),
        "union_size": len(union),
        "only_in_a": sorted(names_a - names_b),
        "only_in_b": sorted(names_b - names_a),
        "common": sorted(intersection),
        "near_matches": near_matches_a_to_b,
    }


def type_agreement(fields_a: list[dict], fields_b: list[dict]) -> float:
    """For overlapping fields, how often do the types match?"""
    map_a = {f["name"]: f["type"] for f in fields_a}
    map_b = {f["name"]: f["type"] for f in fields_b}
    common = set(map_a.keys()) & set(map_b.keys())
    if not common:
        return 0.0
    agree = sum(1 for n in common if map_a[n] == map_b[n])
    return round(agree / len(common), 3)


def extraction_quality(
    client, deployment: str,
    cluster_label: str,
    fields: list[dict],
    sample_chunks: list[str],
    config: FieldDiscoveryConfig,
) -> dict:
    """Run extraction on a small sample and compute quality signals."""
    results = []
    per_chunk = []
    for chunk_text in sample_chunks[:5]:
        ext = extract_fields_from_chunk(
            client, deployment, cluster_label, chunk_text, fields, config,
        )
        results.append(ext)

    if not results:
        return {"fill_rate": 0, "avg_confidence": 0, "grounded_rate": 0, "n_samples": 0}

    field_names = {f["name"] for f in fields}
    total_slots = len(field_names) * len(results)
    filled = sum(
        sum(1 for fn in field_names if fn in ext)
        for ext in results
    )
    confidences = [
        conf
        for ext in results
        for _, (_, conf) in ext.items()
    ]
    # Grounded = confidence ≥ 0.65 (verbatim or partial match in source)
    grounded = sum(1 for c in confidences if c >= 0.65)
    high_conf = sum(1 for c in confidences if c >= 0.80)

    return {
        "fill_rate": round(filled / total_slots, 3) if total_slots else 0,
        "avg_confidence": round(np.mean(confidences), 3) if confidences else 0,
        "median_confidence": round(np.median(confidences), 3) if confidences else 0,
        "grounded_rate": round(grounded / len(confidences), 3) if confidences else 0,
        "high_conf_rate": round(high_conf / len(confidences), 3) if confidences else 0,
        "n_samples": len(results),
        "n_extractions": filled,
        "n_total_slots": total_slots,
    }


# ── Main comparison ──────────────────────────────────────────────────────────

def run_comparison(db_path: str, n_clusters: int = 5, skip_extraction: bool = False):
    conn = duckdb.connect(db_path, read_only=True)
    config = FieldDiscoveryConfig()

    client = _get_azure_client()
    deployment = _get_deployment(config)
    logger.info(f"Azure deployment: {deployment}")

    # Get clusters sorted by chunk_count, take a stratified sample
    all_clusters = conn.execute("""
        SELECT c.cluster_id, c.label, c.keywords, c.chunk_count
        FROM clusters c
        WHERE c.chunk_count >= 10
        ORDER BY c.chunk_count DESC
    """).fetchdf().to_dict("records")

    logger.info(f"Found {len(all_clusters)} clusters with ≥10 chunks")

    # Stratified sampling
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

    logger.info(f"Sampled {len(sample)} clusters for comparison")
    print(f"\n{'='*90}")
    print(f" VANILLA RLM vs RLM + LATENT BRIEFING — Field Discovery & Extraction Comparison")
    print(f"{'='*90}")
    print(f" Database: {db_path}")
    print(f" Clusters: {len(sample)} (stratified from {total})")
    print(f" Extraction: {'Yes (5 chunks/cluster)' if not skip_extraction else 'Skipped'}")
    print(f"{'='*90}\n")

    results = []

    for ci, cluster in enumerate(sample):
        cluster_id = cluster["cluster_id"]
        cluster_label = cluster["label"] or "Unknown"
        chunk_count = cluster["chunk_count"]

        try:
            keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(cluster.get("keywords"), str) else (cluster.get("keywords") or [])
        except json.JSONDecodeError:
            keywords = []

        # Fetch chunks
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

        logger.info(f"\n[{ci+1}/{len(sample)}] ── {cluster_label} ({len(chunk_texts)} chunks) ──")

        # ── Vanilla RLM ──
        logger.info(f"  Running Vanilla RLM...")
        t0 = time.time()
        try:
            fields_rlm = discover_fields_for_cluster_rlm(
                cluster_label, keywords, chunk_texts, config,
            )
        except Exception as e:
            logger.warning(f"  Vanilla RLM failed: {e}")
            fields_rlm = []
        time_rlm = round(time.time() - t0, 2)
        logger.info(f"  Vanilla RLM: {len(fields_rlm)} fields in {time_rlm}s")

        # ── RLM + Latent Briefing ──
        logger.info(f"  Running RLM + Latent Briefing...")
        t0 = time.time()
        try:
            fields_lb, briefing_meta = discover_fields_for_cluster_rlm_lb(
                cluster_label, keywords, chunk_texts, config,
                briefing_tau=1.5,
                briefing_max_tokens=2000,
                adaptive_tau=True,
            )
        except Exception as e:
            logger.warning(f"  RLM+LB failed: {e}")
            fields_lb = []
            briefing_meta = {"error": str(e)}
        time_lb = round(time.time() - t0, 2)
        logger.info(f"  RLM+LB:     {len(fields_lb)} fields in {time_lb}s")

        # ── Overlap analysis ──
        overlap = field_overlap(fields_rlm, fields_lb)
        t_agree = type_agreement(fields_rlm, fields_lb)

        # ── Extraction quality (optional) ──
        ext_rlm = ext_lb = None
        if not skip_extraction and (fields_rlm or fields_lb):
            sample_texts = chunk_texts[:5]
            logger.info(f"  Extracting on {len(sample_texts)} sample chunks...")

            if fields_rlm:
                try:
                    ext_rlm = extraction_quality(
                        client, deployment, cluster_label, fields_rlm, sample_texts, config,
                    )
                    logger.info(f"    Vanilla RLM extraction: fill={ext_rlm['fill_rate']}, conf={ext_rlm['avg_confidence']}, grounded={ext_rlm['grounded_rate']}")
                except Exception as e:
                    logger.warning(f"    Vanilla RLM extraction failed: {e}")

            if fields_lb:
                try:
                    ext_lb = extraction_quality(
                        client, deployment, cluster_label, fields_lb, sample_texts, config,
                    )
                    logger.info(f"    RLM+LB extraction:     fill={ext_lb['fill_rate']}, conf={ext_lb['avg_confidence']}, grounded={ext_lb['grounded_rate']}")
                except Exception as e:
                    logger.warning(f"    RLM+LB extraction failed: {e}")

        entry = {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "chunk_count": chunk_count,
            "n_chunks_used": len(chunk_texts),
            "vanilla_rlm": {
                "n_fields": len(fields_rlm),
                "fields": fields_rlm,
                "time_s": time_rlm,
                "extraction": ext_rlm,
            },
            "rlm_lb": {
                "n_fields": len(fields_lb),
                "fields": fields_lb,
                "time_s": time_lb,
                "extraction": ext_lb,
                "briefing_meta": briefing_meta,
            },
            "overlap": overlap,
            "type_agreement": t_agree,
        }
        results.append(entry)

        logger.info(f"  Overlap: jaccard={overlap['jaccard']}, common={overlap['exact_overlap']}, "
                     f"type_agree={t_agree}")

        # Print briefing stats inline
        if briefing_meta.get("n_compactions", 0) > 0:
            logger.info(f"  Briefing: {briefing_meta['n_compactions']} compactions, "
                         f"avg_compression={briefing_meta.get('avg_compression_ratio', 0):.2f}x, "
                         f"tokens_saved={briefing_meta.get('total_tokens_saved', 0)}")

    conn.close()

    # ── Aggregate + print ──
    summary = compute_summary(results)
    output = {"summary": summary, "per_cluster": results}

    out_path = Path(db_path).parent / "rlm_vs_rlm_lb_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    print_summary(summary, results)
    return output


def compute_summary(results: list[dict]) -> dict:
    if not results:
        return {}

    n = len(results)
    rlm_fields = [r["vanilla_rlm"]["n_fields"] for r in results]
    lb_fields = [r["rlm_lb"]["n_fields"] for r in results]
    jaccards = [r["overlap"]["jaccard"] for r in results]
    type_agrees = [r["type_agreement"] for r in results]
    rlm_times = [r["vanilla_rlm"]["time_s"] for r in results]
    lb_times = [r["rlm_lb"]["time_s"] for r in results]

    # Extraction metrics
    rlm_fill = [r["vanilla_rlm"]["extraction"]["fill_rate"] for r in results
                if r["vanilla_rlm"].get("extraction")]
    lb_fill = [r["rlm_lb"]["extraction"]["fill_rate"] for r in results
               if r["rlm_lb"].get("extraction")]
    rlm_conf = [r["vanilla_rlm"]["extraction"]["avg_confidence"] for r in results
                if r["vanilla_rlm"].get("extraction")]
    lb_conf = [r["rlm_lb"]["extraction"]["avg_confidence"] for r in results
               if r["rlm_lb"].get("extraction")]
    rlm_grounded = [r["vanilla_rlm"]["extraction"]["grounded_rate"] for r in results
                    if r["vanilla_rlm"].get("extraction")]
    lb_grounded = [r["rlm_lb"]["extraction"]["grounded_rate"] for r in results
                   if r["rlm_lb"].get("extraction")]
    rlm_high = [r["vanilla_rlm"]["extraction"]["high_conf_rate"] for r in results
                if r["vanilla_rlm"].get("extraction")]
    lb_high = [r["rlm_lb"]["extraction"]["high_conf_rate"] for r in results
               if r["rlm_lb"].get("extraction")]

    # Failure rates
    rlm_failures = sum(1 for r in results if r["vanilla_rlm"]["n_fields"] == 0)
    lb_failures = sum(1 for r in results if r["rlm_lb"]["n_fields"] == 0)

    # Briefing stats
    briefing_compactions = []
    briefing_compression = []
    briefing_tokens_saved = []
    for r in results:
        meta = r["rlm_lb"].get("briefing_meta", {})
        if meta.get("n_compactions", 0) > 0:
            briefing_compactions.append(meta["n_compactions"])
            briefing_compression.append(meta.get("avg_compression_ratio", 0))
            briefing_tokens_saved.append(meta.get("total_tokens_saved", 0))

    return {
        "n_clusters": n,
        "field_counts": {
            "vanilla_rlm_mean": round(np.mean(rlm_fields), 1),
            "vanilla_rlm_median": round(np.median(rlm_fields), 1),
            "rlm_lb_mean": round(np.mean(lb_fields), 1),
            "rlm_lb_median": round(np.median(lb_fields), 1),
            "delta_mean": round(np.mean(lb_fields) - np.mean(rlm_fields), 1),
        },
        "overlap": {
            "mean_jaccard": round(np.mean(jaccards), 3),
            "median_jaccard": round(np.median(jaccards), 3),
            "mean_type_agreement": round(np.mean(type_agrees), 3),
        },
        "latency_s": {
            "vanilla_rlm_mean": round(np.mean(rlm_times), 1),
            "vanilla_rlm_total": round(sum(rlm_times), 1),
            "rlm_lb_mean": round(np.mean(lb_times), 1),
            "rlm_lb_total": round(sum(lb_times), 1),
            "overhead_pct": round((np.mean(lb_times) - np.mean(rlm_times)) / np.mean(rlm_times) * 100, 1) if np.mean(rlm_times) > 0 else 0,
        },
        "extraction": {
            "vanilla_rlm_fill_rate": round(np.mean(rlm_fill), 3) if rlm_fill else None,
            "rlm_lb_fill_rate": round(np.mean(lb_fill), 3) if lb_fill else None,
            "vanilla_rlm_avg_confidence": round(np.mean(rlm_conf), 3) if rlm_conf else None,
            "rlm_lb_avg_confidence": round(np.mean(lb_conf), 3) if lb_conf else None,
            "vanilla_rlm_grounded_rate": round(np.mean(rlm_grounded), 3) if rlm_grounded else None,
            "rlm_lb_grounded_rate": round(np.mean(lb_grounded), 3) if lb_grounded else None,
            "vanilla_rlm_high_conf_rate": round(np.mean(rlm_high), 3) if rlm_high else None,
            "rlm_lb_high_conf_rate": round(np.mean(lb_high), 3) if lb_high else None,
        },
        "reliability": {
            "vanilla_rlm_failures": rlm_failures,
            "rlm_lb_failures": lb_failures,
        },
        "latent_briefing": {
            "clusters_with_compaction": len(briefing_compactions),
            "avg_compactions_per_cluster": round(np.mean(briefing_compactions), 1) if briefing_compactions else 0,
            "avg_compression_ratio": round(np.mean(briefing_compression), 2) if briefing_compression else 0,
            "total_tokens_saved": int(sum(briefing_tokens_saved)) if briefing_tokens_saved else 0,
        },
    }


def print_summary(summary: dict, results: list[dict]):
    """Pretty-print the comparison results."""
    W = 90
    print(f"\n{'='*W}")
    print(f" RESULTS: Vanilla RLM vs RLM + Latent Briefing")
    print(f"{'='*W}")

    fc = summary.get("field_counts", {})
    lat = summary.get("latency_s", {})
    rel = summary.get("reliability", {})
    ext = summary.get("extraction", {})
    lb = summary.get("latent_briefing", {})

    print(f"\n┌{'─'*44}┬{'─'*20}┬{'─'*20}┐")
    print(f"│ {'Metric':<42} │ {'Vanilla RLM':>18} │ {'RLM + LB':>18} │")
    print(f"├{'─'*44}┼{'─'*20}┼{'─'*20}┤")

    # Field counts
    print(f"│ {'FIELD DISCOVERY':<42} │ {'':>18} │ {'':>18} │")
    print(f"│ {'  Mean fields/cluster':<42} │ {fc.get('vanilla_rlm_mean', 'N/A'):>18} │ {fc.get('rlm_lb_mean', 'N/A'):>18} │")
    print(f"│ {'  Median fields/cluster':<42} │ {fc.get('vanilla_rlm_median', 'N/A'):>18} │ {fc.get('rlm_lb_median', 'N/A'):>18} │")
    delta = fc.get('delta_mean', 0)
    delta_str = f"{'+' if delta >= 0 else ''}{delta}"
    print(f"│ {'  Delta (LB - Vanilla)':<42} │ {'':>18} │ {delta_str:>18} │")

    # Reliability
    print(f"│ {'  Failures (0 fields)':<42} │ {rel.get('vanilla_rlm_failures', 'N/A'):>18} │ {rel.get('rlm_lb_failures', 'N/A'):>18} │")

    # Latency
    print(f"├{'─'*44}┼{'─'*20}┼{'─'*20}┤")
    print(f"│ {'LATENCY':<42} │ {'':>18} │ {'':>18} │")
    print(f"│ {'  Mean time/cluster (s)':<42} │ {lat.get('vanilla_rlm_mean', 'N/A'):>18} │ {lat.get('rlm_lb_mean', 'N/A'):>18} │")
    print(f"│ {'  Total time (s)':<42} │ {lat.get('vanilla_rlm_total', 'N/A'):>18} │ {lat.get('rlm_lb_total', 'N/A'):>18} │")
    overhead = lat.get('overhead_pct', 0)
    print(f"│ {'  LB overhead':<42} │ {'':>18} │ {f'{overhead:+.1f}%':>18} │")

    # Extraction (if available)
    if ext.get("vanilla_rlm_fill_rate") is not None:
        print(f"├{'─'*44}┼{'─'*20}┼{'─'*20}┤")
        print(f"│ {'EXTRACTION QUALITY':<42} │ {'':>18} │ {'':>18} │")
        print(f"│ {'  Fill rate':<42} │ {ext.get('vanilla_rlm_fill_rate', 'N/A'):>18} │ {ext.get('rlm_lb_fill_rate', 'N/A'):>18} │")
        print(f"│ {'  Avg confidence':<42} │ {ext.get('vanilla_rlm_avg_confidence', 'N/A'):>18} │ {ext.get('rlm_lb_avg_confidence', 'N/A'):>18} │")
        print(f"│ {'  Grounded rate (conf≥0.65)':<42} │ {ext.get('vanilla_rlm_grounded_rate', 'N/A'):>18} │ {ext.get('rlm_lb_grounded_rate', 'N/A'):>18} │")
        print(f"│ {'  High-conf rate (conf≥0.80)':<42} │ {ext.get('vanilla_rlm_high_conf_rate', 'N/A'):>18} │ {ext.get('rlm_lb_high_conf_rate', 'N/A'):>18} │")

    # Overlap between the two approaches
    ov = summary.get("overlap", {})
    print(f"├{'─'*44}┼{'─'*20}┼{'─'*20}┤")
    print(f"│ {'FIELD OVERLAP (Vanilla ∩ LB)':<42} │ {'':>18} │ {'':>18} │")
    print(f"│ {'  Mean Jaccard index':<42} │ {ov.get('mean_jaccard', 'N/A'):>38} │")
    print(f"│ {'  Type agreement (common fields)':<42} │ {ov.get('mean_type_agreement', 'N/A'):>38} │")

    # Latent briefing stats
    print(f"├{'─'*44}┼{'─'*20}┼{'─'*20}┤")
    print(f"│ {'LATENT BRIEFING STATS':<42} │ {'':>18} │ {'':>18} │")
    print(f"│ {'  Clusters with compaction':<42} │ {'':>18} │ {lb.get('clusters_with_compaction', 0):>18} │")
    print(f"│ {'  Avg compactions/cluster':<42} │ {'':>18} │ {lb.get('avg_compactions_per_cluster', 0):>18} │")
    cr = lb.get('avg_compression_ratio', 0)
    print(f"│ {'  Avg compression ratio':<42} │ {'':>18} │ {f'{cr:.2f}x':>18} │")
    print(f"│ {'  Total tokens saved':<42} │ {'':>18} │ {lb.get('total_tokens_saved', 0):>18} │")

    print(f"└{'─'*44}┴{'─'*20}┴{'─'*20}┘")

    # Per-cluster detail table
    print(f"\n{'─'*W}")
    print(f" Per-Cluster Details")
    print(f"{'─'*W}")
    header = f"{'Cluster':<35} {'Chunks':>6} │ {'RLM':>4} {'LB':>4} {'Jacc':>6} │"
    if not all(r["vanilla_rlm"].get("extraction") is None for r in results):
        header += f" {'R.Fill':>6} {'L.Fill':>6} {'R.Conf':>6} {'L.Conf':>6}"
    print(header)
    print(f"{'─'*W}")

    for r in results:
        label = r["cluster_label"][:33]
        line = (f"{label:<35} {r['chunk_count']:>6} │ "
                f"{r['vanilla_rlm']['n_fields']:>4} "
                f"{r['rlm_lb']['n_fields']:>4} "
                f"{r['overlap']['jaccard']:>6.3f} │")

        ext_rlm = r["vanilla_rlm"].get("extraction")
        ext_lb = r["rlm_lb"].get("extraction")
        if ext_rlm or ext_lb:
            rf = ext_rlm["fill_rate"] if ext_rlm else 0
            lf = ext_lb["fill_rate"] if ext_lb else 0
            rc = ext_rlm["avg_confidence"] if ext_rlm else 0
            lc = ext_lb["avg_confidence"] if ext_lb else 0
            line += f" {rf:>6.3f} {lf:>6.3f} {rc:>6.3f} {lc:>6.3f}"

        print(line)

    # Field-level detail for each cluster
    print(f"\n{'─'*W}")
    print(f" Field-Level Analysis (per cluster)")
    print(f"{'─'*W}")
    for r in results:
        label = r["cluster_label"]
        rlm_names = sorted(f["name"] for f in r["vanilla_rlm"]["fields"])
        lb_names = sorted(f["name"] for f in r["rlm_lb"]["fields"])
        rlm_set = set(rlm_names)
        lb_set = set(lb_names)
        common = sorted(rlm_set & lb_set)
        only_rlm = sorted(rlm_set - lb_set)
        only_lb = sorted(lb_set - rlm_set)

        print(f"\n  {label}")
        print(f"    Common ({len(common)}):     {', '.join(common) if common else '—'}")
        print(f"    Only Vanilla ({len(only_rlm)}): {', '.join(only_rlm) if only_rlm else '—'}")
        print(f"    Only LB ({len(only_lb)}):      {', '.join(only_lb) if only_lb else '—'}")

    print(f"\n{'='*W}")
    print(f" Comparison complete. JSON saved to data/rlm_vs_rlm_lb_comparison.json")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Vanilla RLM vs RLM + Latent Briefing")
    parser.add_argument("db_path", help="Path to DuckDB file")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters (default: 5)")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip extraction quality (faster)")
    args = parser.parse_args()

    run_comparison(args.db_path, n_clusters=args.n_clusters, skip_extraction=args.skip_extraction)
