#!/usr/bin/env python3
"""Performance run: full Public Contracts dataset with detailed timing analysis.

Runs the complete pipeline on ~3000 public contract PDFs, captures per-stage
timing, and generates a comprehensive performance + quality report at the end.

Usage:
    python scripts/run_public_contracts_perf.py 2>&1 | tee data/public_contracts_perf.log
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.minimal_pipeline import MinimalPipelineConfig, run_minimal_pipeline

logger = logging.getLogger(__name__)

PDF_DIR = "/home/azureuser/contract_datasets/public_contracts/Public_Contracts"
RUN_TAG = "public_contracts_perf"
DATA_DIR = PROJECT_ROOT / "data"
REPORT_PATH = DATA_DIR / "public_contracts_perf_report.txt"


def build_config() -> MinimalPipelineConfig:
    return MinimalPipelineConfig(
        device="cuda",
        db_path=str(DATA_DIR / f"{RUN_TAG}.duckdb"),
        lance_path=str(DATA_DIR / f"{RUN_TAG}.lance"),
        artifacts_dir=str(DATA_DIR / "artifacts" / RUN_TAG),
        max_docs=None,  # ALL docs
        use_optuna=True,
        optuna_trials=30,
        evoc_noise_level=0.3,
        discovery_mode="rlm_v3_unified",
        max_sample_chunks=30,
        max_fields_per_cluster=15,
        min_cluster_chunks=20,          # skip tiny clusters — only meaningful ones
        discovery_temperature=0.2,
        extract_values=True,
        min_extraction_confidence=0.5,
        max_llm_concurrency=30,   # more within-cluster parallelism; Stage 4 is API-bound not GPU-bound
        enable_confidence_gate=True,
        confidence_gate_threshold=0.60,
        cochran_sample_fraction=0.3,
        cochran_min_sample=5,
        cochran_max_sample=30,
        enable_soft_membership=True,
        soft_membership_k=3,
        low_confidence_threshold=0.40,
        enable_field_features=True,
        legacy_fields_dir=None,
        enable_feedback=True,
        tenant_id=None,
        # ── Stage 4 optimisations ───────────────────────────────────────
        # Resume from DB: skip Stages 1-3 (already cached in DuckDB)
        resume_from_stage=4,
        # Merge near-duplicate clusters (cosine >= 0.88) into super-clusters
        # so ONE RLM session covers all variants (e.g. 20x "Software License")
        cluster_merge_threshold=0.88,
        # Hard cap: process top-N super-clusters by chunk count
        # With 8936 raw clusters → merge → expect ~200-400 super-clusters
        # then cap at 200 for predictable runtime (~2-4 hours at 5 concurrency)
        max_clusters=200,
    )


def generate_report(result: dict, wall_time: float, db_path: str):
    """Generate a detailed performance and quality report."""
    import duckdb
    import numpy as np

    lines = []
    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 90)
    p("  PUBLIC CONTRACTS — FULL PERFORMANCE & QUALITY REPORT")
    p(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("=" * 90)

    # ── 1. Run Overview ──
    n_docs = result.get("n_documents", 0)
    n_clusters = result.get("n_clusters", 0)
    n_domains = result.get("n_domains", 0)
    n_fields = result.get("n_fields", 0)
    n_extractions = result.get("n_extractions", 0)
    n_rlm_intents = result.get("n_rlm_intents", 0)
    n_profiles = result.get("n_cluster_profiles", 0)

    p()
    p("━" * 90)
    p("  1. RUN OVERVIEW")
    p("━" * 90)
    p(f"  Documents processed:      {n_docs}")
    p(f"  Clusters formed:          {n_clusters}")
    p(f"  Domains (macro):          {n_domains}")
    p(f"  Fields discovered:        {n_fields}")
    p(f"  Extractions produced:     {n_extractions}")
    p(f"  RLM intents discovered:   {n_rlm_intents}")
    p(f"  Cluster profiles built:   {n_profiles}")
    p(f"  Total wall time:          {wall_time:.1f}s ({wall_time/60:.1f} min)")

    # ── 2. Per-Stage Timing ──
    timings = result.get("stage_timings", {})
    p()
    p("━" * 90)
    p("  2. PER-STAGE TIMING")
    p("━" * 90)
    p(f"  {'Stage':<30s} {'Time (s)':>10s} {'% of Total':>12s} {'Per-Doc (s)':>12s}")
    p(f"  {'─' * 30} {'─' * 10} {'─' * 12} {'─' * 12}")

    total_timed = sum(timings.values()) if timings else wall_time
    for stage, secs in timings.items():
        pct = (secs / total_timed * 100) if total_timed > 0 else 0
        per_doc = secs / n_docs if n_docs > 0 else 0
        p(f"  {stage:<30s} {secs:>10.1f} {pct:>11.1f}% {per_doc:>12.2f}")

    p(f"  {'─' * 30} {'─' * 10} {'─' * 12} {'─' * 12}")
    per_doc_total = wall_time / n_docs if n_docs > 0 else 0
    p(f"  {'TOTAL (wall)':<30s} {wall_time:>10.1f} {'100.0':>11s}% {per_doc_total:>12.2f}")

    # ── 3. LLM Stats ──
    llm_stats = result.get("llm_stats", {})
    p()
    p("━" * 90)
    p("  3. LLM PERFORMANCE")
    p("━" * 90)
    total_calls = llm_stats.get("total_calls", 0)
    total_latency = llm_stats.get("total_latency_s", 0)
    avg_latency = llm_stats.get("avg_latency_s", 0)
    p(f"  Total LLM calls:          {total_calls:,}")
    p(f"  Total LLM latency:        {total_latency:,.1f}s ({total_latency/3600:.1f}h)")
    p(f"  Avg latency per call:     {avg_latency:.2f}s")
    if n_docs > 0:
        p(f"  LLM calls per document:   {total_calls / n_docs:.1f}")
    if n_clusters > 0:
        p(f"  LLM calls per cluster:    {total_calls / n_clusters:.1f}")
    if total_calls > 0 and wall_time > 0:
        p(f"  Effective parallelism:     {total_latency / wall_time:.1f}x")
        p(f"  Throughput:               {total_calls / wall_time:.1f} calls/sec")

    # ── 4. Throughput Summary ──
    p()
    p("━" * 90)
    p("  4. THROUGHPUT SUMMARY")
    p("━" * 90)
    if n_docs > 0:
        p(f"  Documents / minute:       {n_docs / (wall_time / 60):.1f}")
        p(f"  Fields / minute:          {n_fields / (wall_time / 60):.1f}")
        p(f"  Extractions / minute:     {n_extractions / (wall_time / 60):.1f}")
    if n_clusters > 0:
        disc_time = timings.get("discovery_extraction", wall_time)
        p(f"  Clusters / minute:        {n_clusters / (disc_time / 60):.1f}")
        p(f"  Avg cluster time:         {disc_time / n_clusters:.1f}s")

    # ── 5. Quality Analysis (from DB) ──
    p()
    p("━" * 90)
    p("  5. EXTRACTION QUALITY ANALYSIS")
    p("━" * 90)

    try:
        db = duckdb.connect(db_path, read_only=True)

        # Confidence distribution
        conf = db.execute("SELECT confidence FROM extractions WHERE confidence IS NOT NULL").fetchnumpy()
        if conf and len(conf.get("confidence", [])) > 0:
            c = np.array(conf["confidence"], dtype=float)
            p(f"  Extraction confidence distribution (n={len(c):,}):")
            p(f"    Mean:     {np.mean(c):.3f}")
            p(f"    Median:   {np.median(c):.3f}")
            p(f"    P25:      {np.percentile(c, 25):.3f}")
            p(f"    P75:      {np.percentile(c, 75):.3f}")
            p(f"    Min:      {np.min(c):.3f}")
            p(f"    Max:      {np.max(c):.3f}")
            p(f"    ≥ 0.9:    {int(np.sum(c >= 0.9)):,} ({np.sum(c >= 0.9)/len(c)*100:.1f}%)")
            p(f"    ≥ 0.7:    {int(np.sum(c >= 0.7)):,} ({np.sum(c >= 0.7)/len(c)*100:.1f}%)")
            p(f"    < 0.5:    {int(np.sum(c < 0.5)):,} ({np.sum(c < 0.5)/len(c)*100:.1f}%)")

        # Coverage
        p()
        coverage = db.execute("""
            SELECT
                (SELECT COUNT(DISTINCT agreement_id) FROM extractions) as docs_with_ext,
                (SELECT COUNT(*) FROM agreements) as total_docs,
                (SELECT COUNT(DISTINCT field_id) FROM extractions) as fields_with_ext,
                (SELECT COUNT(*) FROM field_definitions) as total_fields
        """).fetchone()
        if coverage:
            p(f"  Coverage:")
            p(f"    Docs with extractions:  {coverage[0]:,} / {coverage[1]:,} ({coverage[0]/max(coverage[1],1)*100:.1f}%)")
            p(f"    Fields with values:     {coverage[2]:,} / {coverage[3]:,} ({coverage[2]/max(coverage[3],1)*100:.1f}%)")

        # Per-document extraction density
        p()
        density = db.execute("""
            SELECT agreement_id, COUNT(*) as n_ext
            FROM extractions GROUP BY agreement_id
        """).fetchnumpy()
        if density and len(density.get("n_ext", [])) > 0:
            d = np.array(density["n_ext"], dtype=float)
            p(f"  Extractions per document:")
            p(f"    Mean:     {np.mean(d):.1f}")
            p(f"    Median:   {np.median(d):.1f}")
            p(f"    Min:      {int(np.min(d))}")
            p(f"    Max:      {int(np.max(d))}")

        # Field type distribution
        p()
        ftypes = db.execute("""
            SELECT field_type, COUNT(*) as cnt
            FROM field_definitions
            GROUP BY field_type
            ORDER BY cnt DESC
        """).fetchall()
        if ftypes:
            p(f"  Field type distribution:")
            for ft, cnt in ftypes:
                p(f"    {ft:<20s} {cnt:>6,}")

        # Cluster size distribution
        p()
        csizes = db.execute("SELECT chunk_count FROM clusters WHERE chunk_count > 0").fetchnumpy()
        if csizes and len(csizes.get("chunk_count", [])) > 0:
            s = np.array(csizes["chunk_count"], dtype=float)
            p(f"  Cluster sizes (chunks):")
            p(f"    Mean:     {np.mean(s):.1f}")
            p(f"    Median:   {np.median(s):.1f}")
            p(f"    Min:      {int(np.min(s))}")
            p(f"    Max:      {int(np.max(s))}")

        # Domain distribution
        p()
        domains = db.execute("""
            SELECT d.label, d.agreement_count,
                   COUNT(c.cluster_id) as n_clusters
            FROM domains d
            LEFT JOIN clusters c ON c.domain_id = d.domain_id
            GROUP BY d.domain_id, d.label, d.agreement_count
            ORDER BY d.agreement_count DESC
            LIMIT 15
        """).fetchall()
        if domains:
            p(f"  Top domains:")
            p(f"    {'Domain':<50s} {'Docs':>6s} {'Clusters':>9s}")
            p(f"    {'─' * 50} {'─' * 6} {'─' * 9}")
            for label, n_agr, n_cl in domains:
                label_trunc = label[:48] if label else "Unknown"
                p(f"    {label_trunc:<50s} {n_agr:>6} {n_cl:>9}")

        # Intent summary
        p()
        intent_stats = db.execute("""
            SELECT
                (SELECT COUNT(*) FROM clause_intents) as total_intents,
                (SELECT COUNT(DISTINCT intent_type_id) FROM clause_intents) as unique_types,
                (SELECT COUNT(DISTINCT clause_id) FROM clause_intents) as clauses_covered
        """).fetchone()
        if intent_stats:
            p(f"  Intent discovery:")
            p(f"    Total intents:         {intent_stats[0]:,}")
            p(f"    Unique intent types:   {intent_stats[1]:,}")
            p(f"    Clauses with intents:  {intent_stats[2]:,}")

        # Top intent types
        top_intents = db.execute("""
            SELECT intent_label, COUNT(*) as cnt
            FROM clause_intents
            GROUP BY intent_label
            ORDER BY cnt DESC
            LIMIT 15
        """).fetchall()
        if top_intents:
            p(f"    Top intent labels:")
            for label, cnt in top_intents:
                p(f"      {label:<40s} {cnt:>6,}")

        # Top extracted fields by frequency
        p()
        top_fields = db.execute("""
            SELECT fd.name, fd.field_type, COUNT(e.extraction_id) as n_ext,
                   AVG(e.confidence) as avg_conf
            FROM field_definitions fd
            JOIN extractions e ON e.field_id = fd.field_id
            GROUP BY fd.field_id, fd.name, fd.field_type
            ORDER BY n_ext DESC
            LIMIT 20
        """).fetchall()
        if top_fields:
            p(f"  Top 20 extracted fields:")
            p(f"    {'Field':<35s} {'Type':<12s} {'Count':>7s} {'Avg Conf':>10s}")
            p(f"    {'─' * 35} {'─' * 12} {'─' * 7} {'─' * 10}")
            for name, ftype, n_ext, avg_conf in top_fields:
                name_trunc = name[:33] if name else "?"
                p(f"    {name_trunc:<35s} {ftype:<12s} {n_ext:>7,} {avg_conf:>10.3f}")

        db.close()
    except Exception as e:
        p(f"  [DB analysis error: {e}]")

    # ── 6. Bottleneck Analysis ──
    p()
    p("━" * 90)
    p("  6. BOTTLENECK ANALYSIS")
    p("━" * 90)
    if timings:
        sorted_stages = sorted(timings.items(), key=lambda x: x[1], reverse=True)
        p(f"  Slowest stage: {sorted_stages[0][0]} ({sorted_stages[0][1]:.0f}s, "
          f"{sorted_stages[0][1]/wall_time*100:.0f}% of wall time)")
        if len(sorted_stages) > 1:
            p(f"  2nd slowest:   {sorted_stages[1][0]} ({sorted_stages[1][1]:.0f}s, "
              f"{sorted_stages[1][1]/wall_time*100:.0f}%)")

    if total_calls > 0:
        llm_wall_pct = total_latency / wall_time * 100 if wall_time > 0 else 0
        p(f"  LLM wall-time utilization: {llm_wall_pct:.0f}% "
          f"(sum of all LLM latency vs wall clock)")
        p(f"  → {'LLM-bound' if llm_wall_pct > 200 else 'Compute/IO-bound'} pipeline")

    if n_docs > 0 and n_extractions > 0:
        p(f"  Extraction yield:         {n_extractions / n_docs:.1f} extractions/doc")
        p(f"  Field discovery yield:    {n_fields / max(n_clusters, 1):.1f} fields/cluster")
        p(f"  Intent yield:             {n_rlm_intents / max(n_clusters, 1):.1f} intents/cluster")

    p()
    p("=" * 90)
    p("  END OF REPORT")
    p("=" * 90)

    # Write report to file
    report_text = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\n[Report saved to {REPORT_PATH}]")

    # Also save raw result JSON
    result_path = DATA_DIR / "public_contracts_perf_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[Raw result saved to {result_path}]")


async def run():
    config = build_config()

    print("=" * 90)
    print(f"  PUBLIC CONTRACTS — FULL PERFORMANCE RUN")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    print(f"  PDF dir:       {PDF_DIR}")
    print(f"  DB:            {config.db_path}")
    print(f"  Discovery:     {config.discovery_mode}")
    print(f"  LLM conc:      {config.max_llm_concurrency}")
    print(f"  Optuna trials: {config.optuna_trials}")
    print(f"  Conf gate:     {config.enable_confidence_gate} (threshold={config.confidence_gate_threshold})")
    print("=" * 90)

    os.environ.setdefault("CLUSTER_CONCURRENCY", "15")   # GPU is 1% utilized in Stage 4 — pure LLM API calls
    os.environ.setdefault("RLM_MAX_ITERATIONS", "15")   # match offline-test defaults (was 25 → 67% extra turns)
    os.environ.setdefault("RLM_MAX_LLM_CALLS", "30")   # match offline-test defaults (was 50 → 67% extra calls)

    t0 = time.time()
    result = await run_minimal_pipeline(
        pdf_dir=PDF_DIR,
        account_id=RUN_TAG,
        config=config,
    )
    wall_time = time.time() - t0

    if "error" in result:
        print(f"\nPIPELINE ERROR: {result['error']}")
        return

    generate_report(result, wall_time, config.db_path)

    # Export DB artifacts
    try:
        from scripts.run_public_contracts import export_db_to_json
        export_dir = str(DATA_DIR / f"{RUN_TAG}_export")
        export_db_to_json(config.db_path, export_dir)
    except Exception as e:
        print(f"[Export failed (non-critical): {e}]")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(run())


if __name__ == "__main__":
    main()
