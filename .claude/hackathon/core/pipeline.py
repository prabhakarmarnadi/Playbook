"""
End-to-end pipeline orchestrator — replaces legacy clustering_executor.py + schematize_handler.py.

Legacy flow (4+ services, 8+ network hops):
  1. fields_processor chunks & embeds docs → writes to AIDB via gRPC
  2. clustering_executor loads from AIDB → BERTopic → saves to Azure Blob
  3. ServiceBus triggers schematize_handler
  4. schematize_handler loads clusters from Blob → calls AIFlow for fields → writes to AIDB

V2 flow (1 script, 0 network hops for compute):
  1. Load/parse docs locally
  2. Chunk + embed with local sentence-transformers
  3. Macro cluster (summary embeddings → BERTopic → LLM label)
  4. Micro cluster per domain (clause embeddings → BERTopic → LLM label)
  5. Merge similar topics (FIELD-310)
  6. Discover fields per cluster (LLM)
  7. Extract fields (LLM)
  8. Store everything in DuckDB

All LLM responses can be cached in DuckDB for demo reliability.
"""
import json
import logging
import os
import uuid
from pathlib import Path

import numpy as np

from config import DB_PATH, SAMPLE_DIR
from core.store import ClusteringStore
from core.embedder import Embedder
from core.llm_client import LLMClient
from core.chunker import semantic_chunk, sliding_window_chunk, get_document_summary, SemanticChunkerConfig
from core.clause_embedder import ClauseEmbedder
from core.macro_clusterer import (
    get_document_summaries, cluster_into_domains,
    label_macro_clusters, build_domain_records,
)
from core.micro_clusterer import (
    cluster_domain_chunks, compute_cluster_quality,
    label_micro_clusters, build_cluster_records,
)
from core.topic_merger import HierarchicalTopicMerger
from core.field_discovery import discover_fields_for_cluster as discover_fields
from core.extractor import extract_conditioned
from core.pdf_parser import parse_pdf, parse_pdf_directory, load_parsed_agreements

logger = logging.getLogger(__name__)


def load_sample_documents(data_dir: str | Path | None = None) -> list[dict]:
    """
    Load sample agreements from the data directory.
    Supports .txt files and the legacy sample_documents.json format.
    """
    data_dir = Path(data_dir) if data_dir else SAMPLE_DIR
    agreements = []

    # Check for legacy JSON format
    json_path = data_dir / "sample_documents.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        for doc in data.get("documents", []):
            agreements.append({
                "agreement_id": doc.get("id", f"agr_{uuid.uuid4().hex[:8]}"),
                "filename": doc.get("filename", doc.get("type", f"doc_{len(agreements)}")),
                "raw_text": doc.get("text", ""),
            })
        return agreements

    # Load .txt files
    for fpath in sorted(data_dir.glob("*.txt")):
        agreements.append({
            "agreement_id": f"agr_{uuid.uuid4().hex[:8]}",
            "filename": fpath.name,
            "raw_text": fpath.read_text(errors="replace"),
        })

    # Load .pdf files via dedicated PDF parser
    for fpath in sorted(data_dir.glob("*.pdf")):
        try:
            parsed = parse_pdf(fpath)
            agreements.append({
                "agreement_id": parsed["agreement_id"],
                "filename": parsed["filename"],
                "raw_text": parsed["raw_text"],
            })
        except Exception as e:
            logger.warning(f"Failed to parse PDF {fpath.name}: {e}")

    return agreements


def run_full_pipeline(
    agreements: list[dict] | None = None,
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    skip_extraction: bool = False,
    callback=None,
) -> ClusteringStore:
    """
    Run the full V2 clustering pipeline end-to-end.

    Replaces: clustering_executor.py pipeline + schematize_handler.py orchestration.
    Zero cloud dependencies. All state stored in DuckDB.

    Args:
        agreements: List of agreement dicts. If None, loads from data_dir.
        data_dir: Path to sample data directory.
        db_path: Path to DuckDB database file.
        skip_extraction: If True, skip LLM field extraction (faster for testing).
        callback: Optional function(stage, message) for progress updates.

    Returns:
        ClusteringStore with all pipeline results.
    """
    def progress(stage, msg):
        logger.info(f"[{stage}] {msg}")
        if callback:
            callback(stage, msg)

    # ── Initialize ─────────────────────────────────────────────────────────
    store = ClusteringStore(db_path or DB_PATH)
    embedder = Embedder.get_instance()
    clause_embedder = ClauseEmbedder(embedder)
    llm = LLMClient()
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    store.start_run(run_id, "init")

    # ── Stage 1: Load documents ────────────────────────────────────────────
    progress("load", "Loading documents...")
    if agreements is None:
        agreements = load_sample_documents(data_dir)
    if not agreements:
        progress("error", "No documents found!")
        return store
    progress("load", f"Loaded {len(agreements)} agreements")

    # ── Stage 2: Semantic chunk documents ──────────────────────────────────
    progress("chunk", "Semantic chunking with MiniLM embeddings...")
    chunker_config = SemanticChunkerConfig()
    all_chunks = []
    for agr in agreements:
        chunks = semantic_chunk(
            agr["raw_text"],
            embed_fn=embedder.embed,
            config=chunker_config,
        )
        for chunk in chunks:
            chunk["agreement_id"] = agr["agreement_id"]
        all_chunks.extend(chunks)

        summary = get_document_summary(agr["raw_text"], agr.get("filename", ""))
        store.insert_agreement(
            agr["agreement_id"], agr.get("filename", ""),
            agr["raw_text"], summary,
        )
    progress("chunk", f"Created {len(all_chunks)} semantic chunks from {len(agreements)} documents")

    # ── Stage 3: Embed (clause-level — FIELD-110) ─────────────────────────
    progress("embed", "Computing clause-level embeddings...")
    chunk_embeddings = clause_embedder.embed_clauses(all_chunks)

    # Store chunks with embeddings
    for i, chunk in enumerate(all_chunks):
        store.insert_chunk(
            chunk["chunk_id"], chunk["agreement_id"], chunk["text"],
            chunk["chunk_index"], chunk_embeddings[i], chunk.get("token_count", 0),
        )
    progress("embed", f"Embedded {len(all_chunks)} clauses ({embedder.dim}-dim)")

    # ── Stage 4: Macro clustering ──────────────────────────────────────────
    progress("macro", "Building field-aware summaries for macro clustering...")
    from core.macro_clusterer import get_field_summaries
    summaries = get_field_summaries(
        agreements, all_chunks, chunk_embeddings,
        embed_fn=embedder.embed,
    )
    topics, summary_embeddings, macro_model = cluster_into_domains(summaries, embedder)

    # Label macro clusters
    progress("macro", "Labeling macro domains...")
    macro_labels = label_macro_clusters(llm, macro_model)
    domain_records = build_domain_records(agreements, topics, summary_embeddings, macro_labels)

    # Store domains and update agreements
    topic_to_domain = {}
    for dr in domain_records:
        store.upsert_domain(
            dr["domain_id"], dr["label"], dr["description"],
            dr["agreement_count"], dr["confidence"], dr.get("centroid"),
        )
        topic_to_domain[dr["topic_id"]] = dr["domain_id"]
        # Update agreement-domain assignments
        for idx in dr["agreement_indices"]:
            agr = agreements[idx]
            conf = float(np.dot(summary_embeddings[idx], dr["centroid"])) if dr.get("centroid") is not None else 0.0
            store.update_agreement_domain(agr["agreement_id"], dr["domain_id"], conf)

    store.update_run(run_id, "macro_complete")
    progress("macro", f"Found {len([d for d in domain_records if d['topic_id'] != -1])} domains")

    # ── Stage 5: Micro clustering per domain ───────────────────────────────
    progress("micro", "Micro-clustering within each domain...")
    merger = HierarchicalTopicMerger()
    all_cluster_records = []

    for dr in domain_records:
        if dr["topic_id"] == -1:
            continue
        domain_id = dr["domain_id"]
        domain_label = dr["label"]

        # Get chunks belonging to this domain
        domain_agreement_ids = set(
            agreements[i]["agreement_id"] for i in dr["agreement_indices"]
        )
        domain_chunk_indices = [
            i for i, c in enumerate(all_chunks)
            if c["agreement_id"] in domain_agreement_ids
        ]

        if len(domain_chunk_indices) < 5:
            progress("micro", f"  Skipping {domain_label} — too few chunks ({len(domain_chunk_indices)})")
            continue

        domain_chunk_texts = [all_chunks[i]["text"] for i in domain_chunk_indices]
        domain_chunk_ids = [all_chunks[i]["chunk_id"] for i in domain_chunk_indices]
        domain_embeddings = chunk_embeddings[domain_chunk_indices]

        # Run BERTopic within domain
        progress("micro", f"  Clustering {domain_label} ({len(domain_chunk_texts)} chunks)...")
        micro_topics, micro_model = cluster_domain_chunks(
            domain_chunk_texts, domain_embeddings,
        )

        # Quality scores
        quality = compute_cluster_quality(domain_embeddings, micro_topics)

        # Label micro clusters
        progress("micro", f"  Labeling {domain_label} clusters...")
        micro_labels = label_micro_clusters(llm, micro_model, domain_label)

        # Build cluster records
        cluster_records = build_cluster_records(
            domain_id, domain_chunk_ids, micro_topics,
            domain_embeddings, micro_labels, quality, micro_model,
        )

        # ── Topic merging (FIELD-310) ──────────────────────────────────
        if len(cluster_records) > 1:
            progress("merge", f"  Merging similar topics in {domain_label}...")
            cluster_records = merger.merge_similar_topics(cluster_records)

        # Store clusters and assignments
        for cr in cluster_records:
            # Count unique agreements
            cluster_chunk_ids = cr.get("chunk_ids", [])
            cluster_agreement_ids = set()
            for cid in cluster_chunk_ids:
                for c in all_chunks:
                    if c["chunk_id"] == cid:
                        cluster_agreement_ids.add(c["agreement_id"])
                        break

            store.upsert_cluster(
                cr["cluster_id"], domain_id, cr["label"], cr.get("description", ""),
                cr.get("keywords", []), cr["chunk_count"],
                len(cluster_agreement_ids), cr.get("quality_score"),
                cr.get("centroid"), cr.get("aliases", []), cr.get("merged_from", []),
            )
            # Store centroid for incremental assignment (FIELD-801)
            if cr.get("centroid") is not None:
                store.save_centroid(cr["cluster_id"], cr["centroid"], cr["chunk_count"])

            # Store assignments
            for cid in cluster_chunk_ids:
                store.insert_cluster_assignment(cid, cr["cluster_id"])

        all_cluster_records.extend(cluster_records)

    store.update_run(run_id, "micro_complete")
    progress("micro", f"Total: {len(all_cluster_records)} clause types across all domains")

    # ── Stage 6: Field discovery ───────────────────────────────────────────
    if not skip_extraction:
        progress("fields", "Discovering fields per cluster...")
        for cr in all_cluster_records:
            # Get representative chunks
            rep_chunks = []
            for cid in cr.get("chunk_ids", [])[:10]:
                for c in all_chunks:
                    if c["chunk_id"] == cid:
                        rep_chunks.append(c["text"])
                        break

            if not rep_chunks:
                continue

            fields = discover_fields(
                llm, cr["label"], cr.get("description", ""), rep_chunks,
            )
            for f in fields:
                store.upsert_field(
                    f["field_id"], cr["cluster_id"], f["name"],
                    f["field_type"], f.get("description", ""), f.get("source", "auto_discovered"),
                )

        store.update_run(run_id, "fields_complete")
        progress("fields", "Field discovery complete")

    # ── Stage 7: Field extraction (example set) ───────────────────────────
    if not skip_extraction:
        progress("extract", "Extracting fields from example sets...")
        for cr in all_cluster_records:
            fields = store.get_fields(cr["cluster_id"])
            if not fields:
                continue

            # Get representative chunks (up to EXAMPLE_SET_SIZE)
            rep_chunk_ids = cr.get("chunk_ids", [])[:20]
            for cid in rep_chunk_ids:
                # Find chunk text
                chunk_text = ""
                chunk_agr = ""
                for c in all_chunks:
                    if c["chunk_id"] == cid:
                        chunk_text = c["text"]
                        chunk_agr = c["agreement_id"]
                        break
                if not chunk_text:
                    continue

                result = extract_conditioned(
                    llm, chunk_text, cr["label"], cr.get("description", ""), fields,
                )
                for f in fields:
                    val_info = result.get(f["name"], {})
                    if val_info.get("value") is not None:
                        store.insert_extraction(
                            f"ext_{uuid.uuid4().hex[:10]}", chunk_agr, f["field_id"],
                            str(val_info["value"]), val_info.get("confidence", 0.5), cid,
                        )

        store.update_run(run_id, "extraction_complete")
        progress("extract", "Extraction complete")

    store.update_run(run_id, "completed", "completed")
    stats = store.get_stats()
    progress("done", f"Pipeline complete: {stats}")
    return store
