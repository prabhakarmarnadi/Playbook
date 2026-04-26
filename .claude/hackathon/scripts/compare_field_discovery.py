#!/usr/bin/env python3
"""
Compare Standard (prompt-based) vs DSPy RLM field discovery on the same clusters.

Usage:
    python scripts/compare_field_discovery.py data/cuad_510_snapshot.duckdb
    python scripts/compare_field_discovery.py data/cuad_510_snapshot.duckdb --n-clusters 30
    python scripts/compare_field_discovery.py data/cuad_510_snapshot.duckdb --n-clusters 10 --skip-extraction
"""
import argparse
import json
import logging
import os
import sys
import time
import uuid
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
    discover_fields_hybrid,
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

    # Semantic overlap: check for fields with similar names (edit distance ≤ 3)
    sem_matches = 0
    for na in names_a:
        for nb in names_b:
            if na == nb:
                continue
            # Simple substring containment check
            if na in nb or nb in na:
                sem_matches += 1
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
        "near_matches": sem_matches,
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
    for chunk_text in sample_chunks[:5]:
        ext = extract_fields_from_chunk(
            client, deployment, cluster_label, chunk_text, fields, config,
        )
        results.append(ext)

    if not results:
        return {"fill_rate": 0, "avg_confidence": 0, "n_samples": 0}

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

    return {
        "fill_rate": round(filled / total_slots, 3) if total_slots else 0,
        "avg_confidence": round(np.mean(confidences), 3) if confidences else 0,
        "n_samples": len(results),
        "n_extractions": filled,
    }


# ── Main comparison ──────────────────────────────────────────────────────────

def run_comparison(db_path: str, n_clusters: int = 20, skip_extraction: bool = False,
                   include_hybrid: bool = False):
    conn = duckdb.connect(db_path, read_only=True)
    config = FieldDiscoveryConfig()

    client = _get_azure_client()
    deployment = _get_deployment(config)
    logger.info(f"Azure deployment: {deployment}")

    # Get clusters sorted by chunk_count, take a stratified sample
    all_clusters = conn.execute("""
        SELECT c.cluster_id, c.label, c.keywords, c.chunk_count, c.domain_id
        FROM clusters c
        WHERE c.chunk_count >= 10
        ORDER BY c.chunk_count DESC
    """).fetchdf().to_dict("records")

    logger.info(f"Found {len(all_clusters)} clusters with ≥10 chunks")

    # Stratified sampling: take from top, middle, bottom
    total = len(all_clusters)
    if total <= n_clusters:
        sample = all_clusters
    else:
        # Top 40%, middle 30%, bottom 30%
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

    results = []

    for ci, cluster in enumerate(sample):
        cluster_id = cluster["cluster_id"]
        cluster_label = cluster["label"] or "Unknown"
        chunk_count = cluster["chunk_count"]

        try:
            keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(cluster.get("keywords"), str) else (cluster.get("keywords") or [])
        except json.JSONDecodeError:
            keywords = []

        # Fetch chunks for this cluster
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

        logger.info(f"[{ci+1}/{len(sample)}] {cluster_label} ({len(chunk_texts)} chunks)")

        # ── Standard field discovery ──
        t0 = time.time()
        try:
            fields_std = discover_fields_for_cluster(
                client, deployment, cluster_label, keywords, chunk_texts, config,
            )
        except Exception as e:
            logger.warning(f"  Standard failed: {e}")
            fields_std = []
        time_std = round(time.time() - t0, 2)

        logger.info(f"  Standard: {len(fields_std)} fields in {time_std}s → {[f['name'] for f in fields_std]}")

        # ── DSPy RLM field discovery ──
        t0 = time.time()
        try:
            fields_rlm = discover_fields_for_cluster_rlm(
                cluster_label, keywords, chunk_texts, config,
            )
        except Exception as e:
            logger.warning(f"  RLM failed: {e}")
            fields_rlm = []
        time_rlm = round(time.time() - t0, 2)

        logger.info(f"  RLM:      {len(fields_rlm)} fields in {time_rlm}s → {[f['name'] for f in fields_rlm]}")

        # ── Overlap analysis ──
        overlap = field_overlap(fields_std, fields_rlm)
        t_agree = type_agreement(fields_std, fields_rlm)

        # ── Hybrid field discovery (Standard + RLM merged) ──
        hybrid_entry = None
        if include_hybrid and fields_std:
            # Re-use already-computed fields instead of calling LLMs again
            from core.field_discovery import _merge_field_lists
            fields_hybrid = _merge_field_lists(fields_std, fields_rlm, config.max_fields_per_cluster)
            time_hybrid = round(time_std + time_rlm, 2)  # wall-clock sum

            ext_hybrid = None
            if not skip_extraction and fields_hybrid:
                sample_texts = chunk_texts[:5]
                try:
                    ext_hybrid = extraction_quality(
                        client, deployment, cluster_label, fields_hybrid, sample_texts, config,
                    )
                except Exception as e:
                    logger.warning(f"  Hybrid extraction failed: {e}")

            source_counts = {}
            for f in fields_hybrid:
                src = f.get("source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1

            hybrid_entry = {
                "n_fields": len(fields_hybrid),
                "fields": fields_hybrid,
                "time_s": time_hybrid,
                "extraction": ext_hybrid,
                "source_breakdown": source_counts,
            }
            logger.info(f"  Hybrid:   {len(fields_hybrid)} fields ({source_counts})")

        # ── Extraction quality (optional) ──
        ext_std = ext_rlm = None
        if not skip_extraction and fields_std and fields_rlm:
            sample_texts = chunk_texts[:5]
            logger.info(f"  Extracting on {len(sample_texts)} sample chunks...")
            try:
                ext_std = extraction_quality(
                    client, deployment, cluster_label, fields_std, sample_texts, config,
                )
            except Exception as e:
                logger.warning(f"  Standard extraction failed: {e}")
            try:
                ext_rlm = extraction_quality(
                    client, deployment, cluster_label, fields_rlm, sample_texts, config,
                )
            except Exception as e:
                logger.warning(f"  RLM extraction failed: {e}")

        entry = {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "chunk_count": chunk_count,
            "n_chunks_used": len(chunk_texts),
            "standard": {
                "n_fields": len(fields_std),
                "fields": fields_std,
                "time_s": time_std,
                "extraction": ext_std,
            },
            "rlm": {
                "n_fields": len(fields_rlm),
                "fields": fields_rlm,
                "time_s": time_rlm,
                "extraction": ext_rlm,
            },
            "hybrid": hybrid_entry,
            "overlap": overlap,
            "type_agreement": t_agree,
        }
        results.append(entry)
        logger.info(f"  Overlap: jaccard={overlap['jaccard']}, common={overlap['exact_overlap']}, "
                     f"type_agree={t_agree}")

    conn.close()

    # ── Aggregate summary ──
    summary = compute_summary(results)
    output = {"summary": summary, "per_cluster": results}

    out_path = Path(db_path).parent / "field_discovery_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    print_summary(summary, results)
    return output


def compute_summary(results: list[dict]) -> dict:
    if not results:
        return {}

    n = len(results)
    std_fields = [r["standard"]["n_fields"] for r in results]
    rlm_fields = [r["rlm"]["n_fields"] for r in results]
    jaccards = [r["overlap"]["jaccard"] for r in results]
    type_agrees = [r["type_agreement"] for r in results]
    std_times = [r["standard"]["time_s"] for r in results]
    rlm_times = [r["rlm"]["time_s"] for r in results]

    # Extraction metrics (where available)
    std_fill = [r["standard"]["extraction"]["fill_rate"] for r in results
                if r["standard"]["extraction"]]
    rlm_fill = [r["rlm"]["extraction"]["fill_rate"] for r in results
                if r["rlm"]["extraction"]]
    std_conf = [r["standard"]["extraction"]["avg_confidence"] for r in results
                if r["standard"]["extraction"]]
    rlm_conf = [r["rlm"]["extraction"]["avg_confidence"] for r in results
                if r["rlm"]["extraction"]]

    # RLM fallback rate (returned 0 fields)
    rlm_failures = sum(1 for r in results if r["rlm"]["n_fields"] == 0)
    std_failures = sum(1 for r in results if r["standard"]["n_fields"] == 0)

    # Hybrid metrics (when available)
    hybrid_stats = {}
    hybrid_results = [r for r in results if r.get("hybrid")]
    if hybrid_results:
        h_fields = [r["hybrid"]["n_fields"] for r in hybrid_results]
        h_times = [r["hybrid"]["time_s"] for r in hybrid_results]
        h_fill = [r["hybrid"]["extraction"]["fill_rate"] for r in hybrid_results
                  if r["hybrid"].get("extraction")]
        h_conf = [r["hybrid"]["extraction"]["avg_confidence"] for r in hybrid_results
                  if r["hybrid"].get("extraction")]
        # Aggregate source breakdown
        all_sources: dict[str, int] = {}
        for r in hybrid_results:
            for src, cnt in r["hybrid"].get("source_breakdown", {}).items():
                all_sources[src] = all_sources.get(src, 0) + cnt
        hybrid_stats = {
            "n_clusters": len(hybrid_results),
            "mean_fields": round(np.mean(h_fields), 1),
            "median_fields": round(np.median(h_fields), 1),
            "mean_time": round(np.mean(h_times), 1),
            "total_time": round(sum(h_times), 1),
            "fill_rate": round(np.mean(h_fill), 3) if h_fill else None,
            "avg_confidence": round(np.mean(h_conf), 3) if h_conf else None,
            "source_breakdown": all_sources,
        }

    return {
        "n_clusters": n,
        "field_counts": {
            "standard_mean": round(np.mean(std_fields), 1),
            "standard_median": round(np.median(std_fields), 1),
            "rlm_mean": round(np.mean(rlm_fields), 1),
            "rlm_median": round(np.median(rlm_fields), 1),
        },
        "overlap": {
            "mean_jaccard": round(np.mean(jaccards), 3),
            "median_jaccard": round(np.median(jaccards), 3),
            "mean_type_agreement": round(np.mean(type_agrees), 3),
        },
        "latency_s": {
            "standard_mean": round(np.mean(std_times), 1),
            "standard_total": round(sum(std_times), 1),
            "rlm_mean": round(np.mean(rlm_times), 1),
            "rlm_total": round(sum(rlm_times), 1),
        },
        "extraction": {
            "standard_fill_rate": round(np.mean(std_fill), 3) if std_fill else None,
            "rlm_fill_rate": round(np.mean(rlm_fill), 3) if rlm_fill else None,
            "standard_avg_confidence": round(np.mean(std_conf), 3) if std_conf else None,
            "rlm_avg_confidence": round(np.mean(rlm_conf), 3) if rlm_conf else None,
        },
        "reliability": {
            "standard_failures": std_failures,
            "rlm_failures": rlm_failures,
            "rlm_fallback_rate": round(rlm_failures / n, 3) if n else 0,
        },
        "hybrid": hybrid_stats,
    }


def print_summary(summary: dict, results: list[dict]):
    """Pretty-print the comparison results."""
    print("\n" + "=" * 80)
    print(" FIELD DISCOVERY COMPARISON: Standard vs DSPy RLM")
    print("=" * 80)

    fc = summary.get("field_counts", {})
    print(f"\n{'Metric':<35} {'Standard':>12} {'DSPy RLM':>12}")
    print("-" * 60)
    print(f"{'Avg fields per cluster':<35} {fc.get('standard_mean', 'N/A'):>12} {fc.get('rlm_mean', 'N/A'):>12}")
    print(f"{'Median fields per cluster':<35} {fc.get('standard_median', 'N/A'):>12} {fc.get('rlm_median', 'N/A'):>12}")

    lat = summary.get("latency_s", {})
    print(f"{'Avg latency per cluster (s)':<35} {lat.get('standard_mean', 'N/A'):>12} {lat.get('rlm_mean', 'N/A'):>12}")
    print(f"{'Total latency (s)':<35} {lat.get('standard_total', 'N/A'):>12} {lat.get('rlm_total', 'N/A'):>12}")

    rel = summary.get("reliability", {})
    print(f"{'Failures (0 fields returned)':<35} {rel.get('standard_failures', 'N/A'):>12} {rel.get('rlm_failures', 'N/A'):>12}")

    ext = summary.get("extraction", {})
    if ext.get("standard_fill_rate") is not None:
        print(f"{'Extraction fill rate':<35} {ext.get('standard_fill_rate', 'N/A'):>12} {ext.get('rlm_fill_rate', 'N/A'):>12}")
        print(f"{'Extraction avg confidence':<35} {ext.get('standard_avg_confidence', 'N/A'):>12} {ext.get('rlm_avg_confidence', 'N/A'):>12}")

    ov = summary.get("overlap", {})
    print(f"\n{'Overlap Metrics':}")
    print(f"  Mean Jaccard index:    {ov.get('mean_jaccard', 'N/A')}")
    print(f"  Median Jaccard index:  {ov.get('median_jaccard', 'N/A')}")
    print(f"  Mean type agreement:   {ov.get('mean_type_agreement', 'N/A')}")

    # Per-cluster mini table
    print(f"\n{'Cluster':<40} {'Std':>4} {'RLM':>4} {'Jacc':>6} {'TyAg':>6} {'Std(s)':>7} {'RLM(s)':>7}")
    print("-" * 80)
    for r in results:
        label = r["cluster_label"][:38]
        print(f"{label:<40} {r['standard']['n_fields']:>4} {r['rlm']['n_fields']:>4} "
              f"{r['overlap']['jaccard']:>6.3f} {r['type_agreement']:>6.3f} "
              f"{r['standard']['time_s']:>7.1f} {r['rlm']['time_s']:>7.1f}")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Standard vs DSPy RLM field discovery")
    parser.add_argument("db_path", help="Path to DuckDB file with clustering results")
    parser.add_argument("--n-clusters", type=int, default=20, help="Number of clusters to compare (default: 20)")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip extraction quality comparison (faster)")
    parser.add_argument("--include-hybrid", action="store_true", help="Include hybrid (merged) field discovery results")
    args = parser.parse_args()

    run_comparison(args.db_path, n_clusters=args.n_clusters,
                   skip_extraction=args.skip_extraction,
                   include_hybrid=args.include_hybrid)
