#!/usr/bin/env python3
"""Full-scale run v3: Late interaction retrieval + wide scope + lazy-persist phantom fix.

Changes from v2:
  1. use_late_interaction=True  — ColBERTv2 MaxSim replaces dense FAISS for retrieval
  2. widen_retrieval_scope=True — retriever sees all agreement chunks, not just cluster-member
  3. Lazy-persist: rag_extractor only persists field_definitions with ≥1 extraction
  4. Drop metrics: tracks why fields are dropped (retrieval_miss, low_confidence, no_extractions)

Reads Stage 1-3 results directly from the cached DuckDB (resume_from_stage=4).

Usage:
    nohup python scripts/run_full_v3.py > data/full_v3.log 2>&1 &
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

RUN_TAG = "full_v3"
DATA_DIR = PROJECT_ROOT / "data"
REPORT_PATH = DATA_DIR / "full_v3_report.txt"
RESULT_PATH = DATA_DIR / "full_v3_result.json"

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
        max_fields_per_cluster=10,
        min_cluster_chunks=10,
        min_domain_chunks=50,
        discovery_temperature=0.2,
        # ── batch small clusters ────────────────────────────────────────
        batch_small_clusters=True,
        small_cluster_min=5,
        # ── extraction ──────────────────────────────────────────────────
        extract_values=True,
        min_extraction_confidence=0.5,
        max_llm_concurrency=30,
        max_rag_retry_rounds=2,
        enable_confidence_gate=True,
        confidence_gate_threshold=0.55,
        cochran_sample_fraction=0.3,
        cochran_min_sample=5,
        cochran_max_sample=30,
        # ── NEW: retriever upgrades ─────────────────────────────────────
        use_late_interaction=True,           # ColBERTv2 MaxSim + BM25
        widen_retrieval_scope=True,          # agreement-wide chunks + cluster boost
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
        max_clusters=None,
    )


def generate_report(result: dict, wall_time: float):
    import duckdb

    lines = []
    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 80)
    p(f"  FULL-V3 RUN — REPORT (late interaction + wide scope + lazy persist)")
    p(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("=" * 80)
    p()
    p(f"  Clusters processed (main):    {result.get('n_clusters', 0)}")
    p(f"  Fields discovered:     {result.get('n_fields', 0)}")
    p(f"  Extractions produced:  {result.get('n_extractions', 0)}")
    p(f"  RLM intents:           {result.get('n_rlm_intents', 0)}")
    p(f"  Profiles built:        {result.get('n_cluster_profiles', 0)}")
    p(f"  Wall time:             {wall_time:.1f}s  ({wall_time/60:.1f} min, {wall_time/3600:.1f} hrs)")
    p()

    # Stage timing
    timings = result.get("stage_timings", {})
    total_t = sum(timings.values()) or 1
    p("  Per-stage timing:")
    for stage, t in sorted(timings.items(), key=lambda x: -x[1]):
        p(f"    {stage:<35s} {t:>8.1f}s  ({t/total_t*100:.0f}%)")
    p()

    # LLM stats
    llm = result.get("llm_stats", {})
    if llm:
        p("  LLM stats:")
        p(f"    Total calls:      {llm.get('total_calls', 0):,}")
        p(f"    Total latency:    {llm.get('total_latency_s', 0):,.1f}s")
        avg = llm.get('total_latency_s', 0) / max(llm.get('total_calls', 1), 1)
        p(f"    Avg per call:     {avg:.2f}s")
    p()

    # Aggregate field drop stats from per-cluster rag_meta
    agg_drops = {
        "fields_persisted": 0,
        "fields_dropped_retrieval_miss": 0,
        "fields_dropped_low_confidence": 0,
        "fields_dropped_no_extractions": 0,
        "fields_already_existed": 0,
    }
    for cs in result.get("cluster_stats", []):
        ds = (cs.get("rag_meta") or {}).get("field_drop_stats", {})
        for k in agg_drops:
            agg_drops[k] += ds.get(k, 0)
    total_dropped = (agg_drops["fields_dropped_retrieval_miss"]
                     + agg_drops["fields_dropped_low_confidence"]
                     + agg_drops["fields_dropped_no_extractions"])
    p("  Lazy-persist field drop stats:")
    p(f"    Fields persisted (had extractions):  {agg_drops['fields_persisted']}")
    p(f"    Fields already existed (agent/seed): {agg_drops['fields_already_existed']}")
    p(f"    Fields DROPPED (total):              {total_dropped}")
    p(f"      - Retrieval miss (no snippets):    {agg_drops['fields_dropped_retrieval_miss']}")
    p(f"      - Low confidence (all null):       {agg_drops['fields_dropped_low_confidence']}")
    p(f"      - No extractions (other):          {agg_drops['fields_dropped_no_extractions']}")
    p()

    # DB stats
    try:
        db = duckdb.connect(SOURCE_DB, read_only=True)
        p("  Extraction quality (from DB):")
        for tbl, label in [
            ("field_definitions", "Field definitions"),
            ("extractions", "Extractions"),
            ("clause_intents", "Clause intents"),
            ("intent_extractions", "Intent extractions"),
        ]:
            n = db.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            p(f"    {label}: {n:,}")

        r = db.execute("SELECT AVG(confidence), MIN(confidence), MAX(confidence) FROM extractions").fetchone()
        if r[0]:
            p(f"    Confidence: avg={r[0]:.3f}, min={r[1]:.3f}, max={r[2]:.3f}")

        r = db.execute("SELECT COUNT(DISTINCT agreement_id) FROM extractions").fetchone()
        total_agr = db.execute("SELECT COUNT(*) FROM agreements").fetchone()[0]
        p(f"    Agreement coverage: {r[0]}/{total_agr} ({r[0]/total_agr*100:.1f}%)")

        # Phantom rate
        r = db.execute("""
            WITH fc AS (
                SELECT fd.field_id, COUNT(e.extraction_id) as n
                FROM field_definitions fd LEFT JOIN extractions e ON e.field_id = fd.field_id
                GROUP BY fd.field_id
            )
            SELECT COUNT(*) as total, SUM(CASE WHEN n=0 THEN 1 ELSE 0 END) as phantom
            FROM fc
        """).fetchone()
        p(f"    Phantom fields: {r[1]}/{r[0]} ({r[1]/r[0]*100:.1f}%)")

        # Phantom by source
        r2 = db.execute("""
            WITH fc AS (
                SELECT fd.field_id, fd.source, COUNT(e.extraction_id) as n
                FROM field_definitions fd LEFT JOIN extractions e ON e.field_id = fd.field_id
                GROUP BY fd.field_id, fd.source
            )
            SELECT source, COUNT(*) as total, SUM(CASE WHEN n=0 THEN 1 ELSE 0 END) as phantom
            FROM fc GROUP BY source ORDER BY total DESC
        """).fetchall()
        p("    Phantom rate by source:")
        for row in r2:
            src, total, phantom = row
            p(f"      {src}: {phantom}/{total} ({phantom/total*100:.1f}%)")

        db.close()
    except Exception as e:
        p(f"  [DB read error: {e}]")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_PATH}")


async def main():
    # ── Clean Stage 4 tables before re-running ──
    import duckdb
    print("Cleaning Stage 4 tables for fresh run...")
    db = duckdb.connect(SOURCE_DB)
    for table in ["field_definitions", "extractions", "clause_intents",
                   "intent_extractions", "domain_field_definitions"]:
        try:
            db.execute(f"DELETE FROM {table}")
            n = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: cleared ({n} remaining)")
        except Exception as e:
            print(f"  {table}: skip ({e})")
    db.close()
    print()

    config = build_config()
    os.environ["CLUSTER_CONCURRENCY"] = "15"

    print(f"Starting full-v3 run at {datetime.now().strftime('%H:%M:%S')}")
    print(f"  min_cluster_chunks={config.min_cluster_chunks}")
    print(f"  batch_small_clusters={config.batch_small_clusters}")
    print(f"  confidence_gate_threshold={config.confidence_gate_threshold}")
    print(f"  use_late_interaction={config.use_late_interaction}")
    print(f"  widen_retrieval_scope={config.widen_retrieval_scope}")
    print()

    t0 = time.time()
    result = await run_minimal_pipeline(
        pdf_dir=PDF_DIR,
        account_id="public_contracts_perf",
        config=config,
    )
    wall = time.time() - t0

    # Save result JSON
    result["config"] = {k: v for k, v in config.__dict__.items()
                        if not k.startswith("_")}
    result["total_elapsed_s"] = round(wall, 1)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    generate_report(result, wall)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
