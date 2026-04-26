#!/usr/bin/env python3
"""Full-scale run: ALL eligible clusters (~458 super-clusters) with quality improvements.

Improvements over mini top-30 run:
  - max_fields_per_cluster: 15 → 10 (reduce phantom fields)
  - Field dedup by embedding cosine similarity (threshold 0.85)
  - Field validation round: quick-sample 3 docs, drop 0/3 fields
  - Post-extraction pruning: delete fields with 0 extractions
  - Type enforcement in RAG extraction prompt
  - Multi-chunk evidence window (adjacent chunk context)
  - Post-extraction type validation (boolean/date/currency)
  - Two-pass domain fields: cross-domain universals seeded into all clusters
  - No max_clusters cap — processes ALL eligible clusters

Reads Stage 1-3 results directly from the cached DuckDB (resume_from_stage=4).

Usage:
    python scripts/run_full_458.py
    python scripts/run_full_458.py 2>&1 | tee data/full_458.log
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

SOURCE_DB = str(PROJECT_ROOT / "data" / "public_contracts_perf.duckdb")
SOURCE_LANCE = str(PROJECT_ROOT / "data" / "public_contracts_perf.lance")

RUN_TAG = "full_458"
DATA_DIR = PROJECT_ROOT / "data"
LOG_PATH = DATA_DIR / "full_458.log"
REPORT_PATH = DATA_DIR / "full_458_report.txt"

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
        max_sample_chunks=30,
        max_fields_per_cluster=10,          # was 15; reduces phantom fields
        min_cluster_chunks=20,
        min_domain_chunks=50,
        discovery_temperature=0.2,
        # ── extraction ──────────────────────────────────────────────────
        extract_values=True,
        min_extraction_confidence=0.5,
        max_llm_concurrency=30,
        max_rag_retry_rounds=2,
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
        resume_from_stage=4,
        cluster_merge_threshold=0.88,
        max_clusters=None,                  # NO CAP — process all clusters
    )


def generate_report(result: dict, wall_time: float):
    import duckdb
    import numpy as np

    lines = []
    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 80)
    p("  FULL-458 RUN — REPORT (with quality improvements)")
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
    p(f"  Wall time:             {wall_time:.1f}s  ({wall_time/60:.1f} min, {wall_time/3600:.1f} hrs)")

    timings = result.get("stage_timings", {})
    if timings:
        p()
        p("  Per-stage timing:")
        total_t = sum(timings.values()) or wall_time
        for stage, secs in timings.items():
            p(f"    {stage:<30s} {secs:>8.1f}s  ({secs/total_t*100:.0f}%)")

    llm = result.get("llm_stats", {})
    if llm.get("total_calls", 0):
        p()
        p("  LLM stats:")
        p(f"    Total calls:      {llm['total_calls']:,}")
        p(f"    Total latency:    {llm.get('total_latency_s', 0):.1f}s")
        p(f"    Avg per call:     {llm.get('avg_latency_s', 0):.2f}s")

    p()
    p("  Extraction quality (from DB):")
    try:
        db = duckdb.connect(SOURCE_DB, read_only=True)
        conf = db.execute(
            "SELECT confidence FROM extractions WHERE confidence IS NOT NULL"
        ).fetchnumpy()
        if conf and len(conf.get("confidence", [])) > 0:
            c = conf["confidence"].astype(float)
            p(f"    n={len(c):,}  mean={c.mean():.3f}  median={float(np.median(c)):.3f}  "
              f"≥0.7={int((c >= 0.7).sum()):,} ({(c >= 0.7).sum() / len(c) * 100:.0f}%)")

        # Field hit rate
        fhr = db.execute("""
            SELECT COUNT(DISTINCT CASE WHEN e.extraction_id IS NOT NULL THEN fd.field_id END)::FLOAT
                   / NULLIF(COUNT(DISTINCT fd.field_id), 0)
            FROM field_definitions fd LEFT JOIN extractions e ON e.field_id = fd.field_id
        """).fetchone()[0]
        p(f"    Field hit rate: {fhr:.0%}" if fhr else "    Field hit rate: N/A")

        # Agreement coverage
        agr_cov = db.execute("""
            SELECT COUNT(DISTINCT agreement_id)::FLOAT /
                   (SELECT COUNT(*) FROM agreements)
            FROM extractions
        """).fetchone()[0]
        p(f"    Agreement coverage: {agr_cov:.1%}" if agr_cov else "    Agreement coverage: N/A")

        # Extractions per agreement
        epa = db.execute("""
            SELECT AVG(n) FROM (
                SELECT COUNT(*) AS n FROM extractions GROUP BY agreement_id
            )
        """).fetchone()[0]
        p(f"    Avg extractions/agreement: {epa:.1f}" if epa else "    Avg extractions/agreement: N/A")

        # Intent stats
        n_intents_db = db.execute("SELECT COUNT(*) FROM clause_intents").fetchone()[0]
        n_ie = db.execute("SELECT COUNT(*) FROM intent_extractions").fetchone()[0]
        p(f"    Intents: {n_intents_db}, Intent extractions: {n_ie}")

        # Domain fields
        n_df = db.execute("SELECT COUNT(*) FROM domain_field_definitions").fetchone()[0]
        p(f"    Domain fields: {n_df}")

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

    result_path = DATA_DIR / "full_458_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[Result → {result_path}]")


async def run():
    config = build_config()

    print("=" * 80)
    print("  FULL-458 RUN (all eligible clusters, quality improvements)")
    print(f"  Started:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Source DB:         {SOURCE_DB}")
    print(f"  max_clusters:      {config.max_clusters or 'None (ALL)'}")
    print(f"  max_fields:        {config.max_fields_per_cluster}")
    print(f"  merge_threshold:   {config.cluster_merge_threshold}")
    print(f"  resume_from_stage: {config.resume_from_stage}")
    print("=" * 80)

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

    generate_report(result, wall_time)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(run())
