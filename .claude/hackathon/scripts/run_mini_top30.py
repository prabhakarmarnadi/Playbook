#!/usr/bin/env python3
"""Mini perf run: top-30 super-clusters, top-30 chunks sampled per cluster.

Reads Stage 1-3 results directly from the cached DuckDB (resume_from_stage=4).
Applies cluster consolidation (merge_threshold=0.88) then caps at 30 super-clusters.
Uses optimised concurrency settings matching offline-test defaults.

Usage:
    python scripts/run_mini_top30.py
    python scripts/run_mini_top30.py 2>&1 | tee data/mini_top30.log
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

# Re-use the same cached DB from the full perf run
SOURCE_DB = str(PROJECT_ROOT / "data" / "public_contracts_perf.duckdb")
SOURCE_LANCE = str(PROJECT_ROOT / "data" / "public_contracts_perf.lance")

RUN_TAG = "mini_top30"
DATA_DIR = PROJECT_ROOT / "data"
LOG_PATH = DATA_DIR / "mini_top30.log"
REPORT_PATH = DATA_DIR / "mini_top30_report.txt"

# PDF dir is required by the signature but skipped when resume_from_stage=4
PDF_DIR = "/home/azureuser/contract_datasets/public_contracts/Public_Contracts"


def build_config() -> MinimalPipelineConfig:
    return MinimalPipelineConfig(
        device="cuda",
        db_path=SOURCE_DB,
        lance_path=SOURCE_LANCE,
        artifacts_dir=str(DATA_DIR / "artifacts" / RUN_TAG),
        max_docs=None,
        use_optuna=True,
        optuna_trials=30,
        evoc_noise_level=0.3,
        # ── discovery ───────────────────────────────────────────────────
        discovery_mode="rlm_v3_unified",
        max_sample_chunks=30,           # top-30 chunks sampled per cluster
        max_fields_per_cluster=15,
        min_cluster_chunks=20,          # skip tiny clusters
        min_domain_chunks=50,           # lower bar for domain-level discovery (Stage 3.5)
        discovery_temperature=0.2,
        # ── extraction ──────────────────────────────────────────────────
        extract_values=True,
        min_extraction_confidence=0.5,
        max_llm_concurrency=30,         # API-bound stage; GPU stays idle
        max_rag_retry_rounds=2,         # cap agentic re-queries (was 3; saves ~33% LLM cost)
        enable_confidence_gate=True,
        confidence_gate_threshold=0.60,
        cochran_sample_fraction=0.3,
        cochran_min_sample=5,
        cochran_max_sample=30,
        # ── soft membership ─────────────────────────────────────────────
        enable_soft_membership=True,
        soft_membership_k=3,
        low_confidence_threshold=0.40,
        enable_field_features=True,
        legacy_fields_dir=None,
        # ── feedback / personalisation ──────────────────────────────────
        enable_feedback=True,
        tenant_id=None,
        # ── Stage 4 optimisations ───────────────────────────────────────
        resume_from_stage=4,            # skip Stages 1-3, read from cached DB
        cluster_merge_threshold=0.88,   # merge near-duplicate clusters
        max_clusters=30,                # process only the top-30 super-clusters
    )


def generate_report(result: dict, wall_time: float):
    import duckdb
    import numpy as np

    lines = []
    def p(s=""):
        lines.append(s)
        print(s)

    db_path = SOURCE_DB

    p("=" * 80)
    p("  MINI TOP-30 RUN — REPORT")
    p(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("=" * 80)

    n_clusters = result.get("n_clusters", 0)
    n_fields   = result.get("n_fields", 0)
    n_ext      = result.get("n_extractions", 0)
    n_intents  = result.get("n_rlm_intents", 0)
    n_profiles = result.get("n_cluster_profiles", 0)

    p()
    p(f"  Clusters processed:    {n_clusters}")
    p(f"  Fields discovered:     {n_fields}")
    p(f"  Extractions produced:  {n_ext}")
    p(f"  RLM intents:           {n_intents}")
    p(f"  Profiles built:        {n_profiles}")
    p(f"  Wall time:             {wall_time:.1f}s  ({wall_time/60:.1f} min)")

    # per-stage timing
    timings = result.get("stage_timings", {})
    if timings:
        p()
        p("  Per-stage timing:")
        total_t = sum(timings.values()) or wall_time
        for stage, secs in timings.items():
            p(f"    {stage:<30s} {secs:>8.1f}s  ({secs/total_t*100:.0f}%)")

    # LLM stats
    llm = result.get("llm_stats", {})
    if llm.get("total_calls", 0):
        p()
        p("  LLM stats:")
        p(f"    Total calls:      {llm['total_calls']:,}")
        p(f"    Total latency:    {llm.get('total_latency_s', 0):.1f}s")
        p(f"    Avg per call:     {llm.get('avg_latency_s', 0):.2f}s")
        p(f"    Calls/cluster:    {llm['total_calls'] / max(n_clusters, 1):.1f}")

    # DB quality snapshot
    p()
    p("  Extraction quality (from DB):")
    try:
        db = duckdb.connect(db_path, read_only=True)
        conf = db.execute(
            "SELECT confidence FROM extractions WHERE confidence IS NOT NULL"
        ).fetchnumpy()
        if conf and len(conf.get("confidence", [])) > 0:
            c = conf["confidence"].astype(float)
            p(f"    n={len(c):,}  mean={c.mean():.3f}  median={float(np.median(c)):.3f}  "
              f"p75={float(np.percentile(c, 75)):.3f}  ≥0.7={int((c >= 0.7).sum()):,}")

        top_fields = db.execute("""
            SELECT fd.name, COUNT(e.extraction_id) as n
            FROM field_definitions fd
            JOIN extractions e ON e.field_id = fd.field_id
            GROUP BY fd.field_id, fd.name
            ORDER BY n DESC LIMIT 15
        """).fetchall()
        if top_fields:
            p()
            p("  Top discovered fields:")
            for name, n in top_fields:
                p(f"    {name:<40s} {n:>5,} extractions")

        db.close()
    except Exception as e:
        p(f"  [DB read error: {e}]")

    p()
    p("=" * 80)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\n[Report → {REPORT_PATH}]")

    result_path = DATA_DIR / "mini_top30_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[Result → {result_path}]")


async def run():
    config = build_config()

    print("=" * 80)
    print("  MINI TOP-30 RUN")
    print(f"  Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Source DB:      {SOURCE_DB}")
    print(f"  max_clusters:   {config.max_clusters}  (top-N super-clusters by chunk count)")
    print(f"  max_sample_chunks: {config.max_sample_chunks}")
    print(f"  merge_threshold:   {config.cluster_merge_threshold}")
    print(f"  resume_from_stage: {config.resume_from_stage}")
    print("=" * 80)

    # Match offline-test defaults; GPU is idle during Stage 4
    os.environ["CLUSTER_CONCURRENCY"] = "15"
    os.environ["RLM_MAX_ITERATIONS"]  = "15"
    os.environ["RLM_MAX_LLM_CALLS"]   = "30"

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

    generate_report(result, wall_time)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(run())


if __name__ == "__main__":
    main()
