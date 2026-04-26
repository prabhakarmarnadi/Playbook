#!/usr/bin/env python3
"""Resume CUAD pipeline from where it crashed (field discovery cluster 72/367).

Picks up remaining clusters, runs field discovery + extraction,
then intent extraction + KG + analytics.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/cuad_510_resume.log"),
    ],
)
logger = logging.getLogger(__name__)

DB_PATH = "data/cuad_510.duckdb"

def progress(stage, msg):
    logger.info(f"[{stage}] {msg}")


def main():
    t0 = time.time()
    from core.store import ClusteringStore
    store = ClusteringStore(DB_PATH)

    # ── 1. Resume field discovery for remaining clusters ──
    logger.info("=" * 60)
    logger.info("  RESUMING: Field discovery for remaining clusters")
    logger.info("=" * 60)

    existing_field_clusters = set(
        r[0] for r in store.conn.execute(
            "SELECT DISTINCT cluster_id FROM field_definitions"
        ).fetchall()
    )
    logger.info(f"Clusters already with fields: {len(existing_field_clusters)}")

    all_clusters = store.conn.execute("""
        SELECT cluster_id, label, chunk_count FROM clusters
        WHERE chunk_count >= 5
        ORDER BY chunk_count DESC
    """).fetchdf().to_dict("records")

    remaining = [c for c in all_clusters if c["cluster_id"] not in existing_field_clusters]
    logger.info(f"Remaining clusters to process: {len(remaining)}/{len(all_clusters)}")

    if remaining:
        from core.field_discovery import (
            FieldDiscoveryConfig, discover_fields_for_cluster,
            extract_fields_from_chunk, _get_azure_client, _get_deployment,
        )
        import uuid
        from concurrent.futures import ThreadPoolExecutor, as_completed

        config = FieldDiscoveryConfig(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini"),
        )
        client = _get_azure_client()
        deployment = _get_deployment(config)

        total_fields = 0
        total_extractions = 0

        for ci, cluster in enumerate(remaining):
            cluster_id = cluster["cluster_id"]
            cluster_label = cluster["label"] or "Unknown"
            try:
                keywords_raw = store.conn.execute(
                    "SELECT keywords FROM clusters WHERE cluster_id = ?", [cluster_id]
                ).fetchone()[0]
                keywords = json.loads(keywords_raw) if keywords_raw else []
            except Exception:
                keywords = []

            chunk_rows = store.conn.execute("""
                SELECT ch.chunk_id, ch.chunk_text, ch.agreement_id
                FROM chunks ch
                JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
                WHERE ca.cluster_id = ?
                ORDER BY ch.chunk_index
            """, [cluster_id]).fetchdf().to_dict("records")

            if len(chunk_rows) < 5:
                continue

            progress("field_discovery",
                     f"  [{ci+1}/{len(remaining)}] {cluster_label} ({len(chunk_rows)} chunks)")

            # Discover fields
            chunk_texts = [r["chunk_text"] for r in chunk_rows]
            try:
                fields = discover_fields_for_cluster(
                    client, deployment, cluster_label, keywords, chunk_texts, config,
                )
            except Exception as e:
                logger.warning(f"Field discovery failed for {cluster_label}: {e}")
                continue

            if not fields:
                continue

            # Persist field definitions
            field_id_map = {}
            for f in fields:
                field_id = f"fld_{uuid.uuid4().hex[:10]}"
                field_id_map[f["name"]] = field_id
                store.upsert_field(
                    field_id=field_id, cluster_id=cluster_id,
                    name=f["name"], field_type=f["type"],
                    description=f["description"], source="auto_discovered",
                    examples=f.get("examples", []),
                )

            total_fields += len(fields)
            progress("field_discovery",
                     f"    {len(fields)} fields: {[f['name'] for f in fields]}")

            # Extract values
            cluster_extractions = 0

            def _extract_one(chunk_row):
                return chunk_row, extract_fields_from_chunk(
                    client, deployment, cluster_label,
                    chunk_row["chunk_text"], fields, config,
                )

            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = {pool.submit(_extract_one, cr): cr for cr in chunk_rows}
                for future in as_completed(futures):
                    try:
                        chunk_row, extractions = future.result()
                    except Exception as e:
                        logger.warning(f"Extraction failed: {e}")
                        continue

                    for field_name, (value, confidence) in extractions.items():
                        if confidence < 0.5:
                            continue
                        field_id = field_id_map.get(field_name)
                        if not field_id or not value:
                            continue
                        store.insert_extraction(
                            extraction_id=f"ext_{uuid.uuid4().hex[:10]}",
                            field_id=field_id,
                            agreement_id=chunk_row["agreement_id"],
                            value=str(value), confidence=confidence,
                            source_chunk_id=chunk_row["chunk_id"],
                        )
                        cluster_extractions += 1

            total_extractions += cluster_extractions
            progress("field_discovery",
                     f"    {cluster_extractions} extractions")

        logger.info(f"Field discovery complete: {total_fields} fields, "
                    f"{total_extractions} extractions in {time.time()-t0:.0f}s")

    # ── 2. Intent extraction ──
    logger.info("=" * 60)
    logger.info("  RUNNING: Intent extraction")
    logger.info("=" * 60)
    try:
        from core.intent_extractor import run_intent_extraction, IntentExtractionConfig
        intent_config = IntentExtractionConfig(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini"),
        )
        intent_result = run_intent_extraction(
            store=store, config=intent_config, progress_fn=progress,
        )
        logger.info(f"Intents: {intent_result}")
    except Exception as e:
        logger.error(f"Intent extraction failed: {e}")

    # ── 3. Knowledge Graph ──
    logger.info("=" * 60)
    logger.info("  RUNNING: Knowledge graph")
    logger.info("=" * 60)
    try:
        from core.knowledge_graph import ContractKnowledgeGraph
        kg = ContractKnowledgeGraph(store)
        kg_stats = kg.build()
        logger.info(f"KG: {kg_stats}")
    except Exception as e:
        logger.error(f"KG failed: {e}")

    # ── 4. Analytics (risk, deviation, normalization, playbook) ──
    logger.info("=" * 60)
    logger.info("  RUNNING: Analytics")
    logger.info("=" * 60)
    try:
        from core.evoc_pipeline import run_analytics
        run_analytics(DB_PATH, use_llm_risk=False, min_instances=10)
    except Exception as e:
        logger.error(f"Analytics failed: {e}")

    # ── 5. Update pipeline run status ──
    store.update_run("run_55633c1c", "completed", "completed")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"  CUAD resume complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info("=" * 60)

    # Final stats
    for table in ['domains', 'agreements', 'chunks', 'clusters',
                   'field_definitions', 'extractions', 'intent_types', 'clause_intents']:
        try:
            count = store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"  {table}: {count}")
        except Exception:
            pass

    store.close()


if __name__ == "__main__":
    main()
