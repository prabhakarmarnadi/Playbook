#!/usr/bin/env python3
"""Run the minimal pipeline on the Public Contracts dataset (~3000 PDFs).

Usage:
    # Sanity check on 15 docs first:
    python scripts/run_public_contracts.py --sanity

    # Full run on all ~3000 docs:
    python scripts/run_public_contracts.py --full

    # Export existing DB to shareable JSON bundle:
    python scripts/run_public_contracts.py --export-only --db data/public_contracts_full.duckdb
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.minimal_pipeline import MinimalPipelineConfig, run_minimal_pipeline
from core.store import ClusteringStore

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
PDF_DIR = "/home/azureuser/contract_datasets/public_contracts/Public_Contracts"
RUN_TAG = "public_contracts"


def get_paths(mode: str) -> dict:
    """Return DB/lance/artifact paths for the given mode."""
    tag = f"{RUN_TAG}_{mode}"
    data_dir = PROJECT_ROOT / "data"
    return {
        "db": str(data_dir / f"{tag}.duckdb"),
        "lance": str(data_dir / f"{tag}.lance"),
        "artifacts": str(data_dir / "artifacts" / tag),
        "tag": tag,
    }


def build_config(mode: str, paths: dict) -> MinimalPipelineConfig:
    """Build pipeline config tuned for the public contracts dataset."""
    is_sanity = mode == "sanity"

    return MinimalPipelineConfig(
        # Core
        device="cuda",
        db_path=paths["db"],
        lance_path=paths["lance"],
        artifacts_dir=paths["artifacts"],

        # Limit docs for sanity check
        max_docs=15 if is_sanity else None,

        # Clustering: full Optuna for full run, reduced for sanity
        use_optuna=True,
        optuna_trials=10 if is_sanity else 30,
        evoc_noise_level=0.3,

        # Discovery: unified mode (discover + inline extract in one pass)
        discovery_mode="rlm_v3_unified",
        max_sample_chunks=30,
        max_fields_per_cluster=15,
        min_cluster_chunks=5,
        discovery_temperature=0.2,

        # Extraction
        extract_values=True,
        min_extraction_confidence=0.5,

        # Concurrency — tuned for throughput
        max_llm_concurrency=12,

        # Confidence gating
        enable_confidence_gate=True,
        confidence_gate_threshold=0.60,
        cochran_sample_fraction=0.3,
        cochran_min_sample=5,
        cochran_max_sample=30,

        # Soft membership
        enable_soft_membership=True,
        soft_membership_k=3,
        low_confidence_threshold=0.40,

        # Field features
        enable_field_features=True,
        legacy_fields_dir=None,

        # Feedback
        enable_feedback=True,
        tenant_id=None,
    )


def export_db_to_json(db_path: str, output_dir: str) -> Path:
    """Export all DB tables to a shareable JSON bundle."""
    import duckdb

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    db = duckdb.connect(db_path, read_only=False)
    t0 = time.time()

    def _query_json(sql: str, filename: str, transform=None):
        df = db.execute(sql).fetchdf()
        records = df.to_dict("records")
        if transform:
            records = [transform(r) for r in records]
        path = out / filename
        with open(path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        logger.info(f"  {filename}: {len(records)} records")
        return len(records)

    counts = {}
    logger.info("Exporting DB tables to JSON...")

    counts["domains"] = _query_json(
        "SELECT domain_id, label, description, agreement_count, confidence FROM domains ORDER BY agreement_count DESC",
        "domains.json",
    )
    counts["clusters"] = _query_json(
        "SELECT cluster_id, domain_id, label, description, keywords, chunk_count FROM clusters ORDER BY chunk_count DESC",
        "clusters.json",
        transform=lambda r: {**r, "keywords": json.loads(r["keywords"]) if isinstance(r.get("keywords"), str) else r.get("keywords", [])},
    )
    counts["field_definitions"] = _query_json(
        "SELECT field_id, cluster_id, name, field_type, description, examples FROM field_definitions ORDER BY cluster_id, name",
        "field_definitions.json",
        transform=lambda r: {**r, "examples": json.loads(r["examples"]) if isinstance(r.get("examples"), str) else r.get("examples", [])},
    )
    counts["extractions"] = _query_json(
        "SELECT extraction_id, agreement_id, field_id, value, confidence, source_chunk_id FROM extractions ORDER BY agreement_id, field_id",
        "extractions.json",
    )
    counts["agreements"] = _query_json(
        "SELECT agreement_id, filename, domain_id, summary_text, domain_confidence FROM agreements ORDER BY filename",
        "agreements.json",
    )
    counts["clauses"] = _query_json(
        "SELECT clause_id, agreement_id, clause_number, clause_title, heading_level, token_count FROM clauses ORDER BY agreement_id, clause_number",
        "clauses_summary.json",
    )
    counts["intent_types"] = _query_json(
        "SELECT intent_type_id, label, description, occurrence_count, aliases FROM intent_types ORDER BY occurrence_count DESC",
        "intent_types.json",
        transform=lambda r: {**r, "aliases": json.loads(r["aliases"]) if isinstance(r.get("aliases"), str) else r.get("aliases", [])},
    )
    counts["clause_intents"] = _query_json(
        "SELECT intent_id, clause_id, agreement_id, clause_type_id, intent_type_id, intent_label, summary, party_from, party_to, attributes FROM clause_intents ORDER BY clause_id",
        "clause_intents.json",
        transform=lambda r: {**r, "attributes": json.loads(r["attributes"]) if isinstance(r.get("attributes"), str) else r.get("attributes", {})},
    )
    counts["cluster_assignments"] = _query_json(
        "SELECT chunk_id, cluster_id, probability FROM cluster_assignments ORDER BY cluster_id, probability DESC",
        "cluster_assignments.json",
    )
    counts["pipeline_runs"] = _query_json(
        "SELECT * FROM pipeline_runs", "pipeline_runs.json",
    )

    stats = db.execute("""
        SELECT
            (SELECT COUNT(*) FROM agreements) as n_agreements,
            (SELECT COUNT(*) FROM chunks) as n_chunks,
            (SELECT COUNT(*) FROM clauses) as n_clauses,
            (SELECT COUNT(*) FROM clusters) as n_clusters,
            (SELECT COUNT(*) FROM domains) as n_domains,
            (SELECT COUNT(*) FROM field_definitions) as n_fields,
            (SELECT COUNT(*) FROM extractions) as n_extractions,
            (SELECT COUNT(*) FROM intent_types) as n_intent_types,
            (SELECT COUNT(*) FROM clause_intents) as n_clause_intents,
            (SELECT AVG(confidence) FROM extractions) as avg_confidence,
            (SELECT COUNT(DISTINCT agreement_id) FROM extractions) as n_agreements_with_extractions
    """).fetchone()

    elapsed = time.time() - t0
    db.close()

    manifest = {
        "export_timestamp": datetime.now().isoformat(),
        "source_db": db_path,
        "dataset": "Public_Contracts",
        "export_time_s": round(elapsed, 1),
        "table_counts": counts,
        "aggregate_stats": {
            "n_agreements": stats[0],
            "n_chunks": stats[1],
            "n_clauses": stats[2],
            "n_clusters": stats[3],
            "n_domains": stats[4],
            "n_fields": stats[5],
            "n_extractions": stats[6],
            "n_intent_types": stats[7],
            "n_clause_intents": stats[8],
            "avg_extraction_confidence": round(stats[9], 3) if stats[9] else None,
            "n_agreements_with_extractions": stats[10],
        },
        "files": sorted(str(p.name) for p in out.glob("*.json")),
    }
    with open(out / "export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Export complete in {elapsed:.1f}s → {out}")
    return out


def export_parquet_backup(db_path: str) -> Path:
    store = ClusteringStore(db_path)
    backup_dir = store.backup()
    store.close()
    logger.info(f"Parquet backup → {backup_dir}")
    return backup_dir


async def run_pipeline(mode: str) -> dict:
    paths = get_paths(mode)
    config = build_config(mode, paths)

    logger.info(f"{'=' * 80}")
    logger.info(f" PUBLIC CONTRACTS PIPELINE: {RUN_TAG}_{mode}")
    logger.info(f"{'=' * 80}")
    logger.info(f"  Mode:       {mode}")
    logger.info(f"  PDF dir:    {PDF_DIR}")
    logger.info(f"  Max docs:   {config.max_docs or 'ALL (~3000)'}")
    logger.info(f"  DB:         {paths['db']}")
    logger.info(f"  Discovery:  {config.discovery_mode}")
    logger.info(f"  LLM concurrency: {config.max_llm_concurrency}")
    logger.info(f"  Confidence gate: {config.enable_confidence_gate} (threshold={config.confidence_gate_threshold})")
    logger.info(f"{'=' * 80}")

    os.environ.setdefault("CLUSTER_CONCURRENCY", "5")
    os.environ.setdefault("RLM_MAX_ITERATIONS", "25")
    os.environ.setdefault("RLM_MAX_LLM_CALLS", "50")

    t0 = time.time()
    result = await run_minimal_pipeline(
        pdf_dir=PDF_DIR,
        account_id=f"{RUN_TAG}_{mode}",
        config=config,
    )
    elapsed = time.time() - t0

    logger.info(f"\n{'=' * 80}")
    logger.info(f" PIPELINE COMPLETE: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    logger.info(f"{'=' * 80}")

    if "error" in result:
        logger.error(f"Pipeline error: {result['error']}")
        return result

    logger.info(f"  Documents:   {result.get('n_documents', '?')}")
    logger.info(f"  Clusters:    {result.get('n_clusters', '?')}")
    logger.info(f"  Fields:      {result.get('n_fields', '?')}")
    logger.info(f"  Extractions: {result.get('n_extractions', '?')}")
    logger.info(f"  RLM Intents: {result.get('n_rlm_intents', '?')}")

    return result


def main():
    parser = argparse.ArgumentParser(description=f"Run pipeline on Public Contracts ({RUN_TAG})")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sanity", action="store_true", help="Run on 15 docs as sanity check")
    group.add_argument("--full", action="store_true", help="Run on all ~3000 docs")
    group.add_argument("--export-only", action="store_true", help="Export existing DB to JSON")
    parser.add_argument("--db", default=None, help="Override DB path (for --export-only)")
    parser.add_argument("--no-export", action="store_true", help="Skip JSON export after pipeline")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)-5s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.export_only:
        db_path = args.db or get_paths("full")["db"]
        if not Path(db_path).exists():
            logger.error(f"DB not found: {db_path}")
            return 1
        export_dir = str(Path(db_path).parent / f"{Path(db_path).stem}_export")
        export_db_to_json(db_path, export_dir)
        export_parquet_backup(db_path)
        return 0

    mode = "sanity" if args.sanity else "full"
    paths = get_paths(mode)

    result = asyncio.run(run_pipeline(mode))
    if "error" in result:
        return 1

    if not args.no_export:
        logger.info("\n" + "=" * 80)
        logger.info(" EXPORTING ARTIFACTS")
        logger.info("=" * 80)
        export_dir = str(Path(paths["db"]).parent / f"{RUN_TAG}_{mode}_export")
        export_db_to_json(paths["db"], export_dir)
        try:
            export_parquet_backup(paths["db"])
        except Exception as e:
            logger.warning(f"Parquet backup failed (non-critical): {e}")

    print(f"\n{'=' * 80}")
    print(f"  RUN COMPLETE: {RUN_TAG}_{mode}")
    print(f"{'=' * 80}")
    print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
