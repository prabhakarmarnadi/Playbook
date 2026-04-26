"""
PDF-to-clusters pipeline — ingest PDFs, chunk, embed, cluster, persist.

A focused pipeline that does exactly:
  1. Parse PDFs (PyMuPDF) → raw text
  2. Chunk text (sliding window)
  3. Embed chunks (Nomic GPU) + document summaries
  4. Macro cluster (document-level domains via cuML UMAP/HDBSCAN)
  5. Micro cluster per domain (clause-level topics)
  6. Persist everything to DuckDB (metadata) + LanceDB (vectors)

No LLM calls, no field discovery, no extraction — just ingest + cluster.
This is the minimal pipeline for getting documents organized.

Usage:
    poetry run python -m core.pdf_pipeline /path/to/pdfs
    poetry run python -m core.pdf_pipeline /path/to/pdfs --db data/my.duckdb --lance data/my.lance
"""
import logging
import sys
import uuid
from pathlib import Path

import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from config import (
    DB_PATH, LEGAL_STOPWORDS, DATA_DIR,
    MACRO_MIN_CLUSTER_SIZE, MACRO_MIN_SAMPLES,
    MICRO_MIN_CLUSTER_SIZE, MICRO_MIN_SAMPLES,
    UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS, UMAP_MIN_DIST,
)
from core.pdf_parser import parse_pdf_directory
from core.chunker import semantic_chunk, get_document_summary, SemanticChunkerConfig
from core.nomic_embedder import NomicEmbedder
from core.cuml_wrappers import make_umap, make_hdbscan, cuml_available
from core.lancedb_store import LanceVectorStore
from core.store import ClusteringStore
from core.topic_merger import HierarchicalTopicMerger
from core.micro_clusterer import compute_cluster_quality

logger = logging.getLogger(__name__)

LANCEDB_PATH = DATA_DIR / "vectors.lance"


def _auto_label_topic(topic_model: BERTopic, topic_id: int, prefix: str = "") -> dict:
    """Generate a label from top keywords — no LLM needed."""
    if topic_id == -1:
        return {"label": "Uncategorized", "description": "Not assigned to any cluster"}
    keywords = [w for w, _ in topic_model.get_topic(topic_id)]
    top = keywords[:5]
    label = f"{prefix}{' / '.join(top[:3]).title()}" if top else f"{prefix}Topic_{topic_id}"
    desc = f"Keywords: {', '.join(top)}" if top else ""
    return {"label": label, "description": desc}


def run_pdf_pipeline(
    pdf_dir: str | Path,
    db_path: str | Path | None = None,
    lance_path: str | Path | None = None,
    parsed_text_dir: str | Path | None = None,
    device: str = "cuda",
    callback=None,
) -> dict:
    """
    Ingest PDFs → chunk → embed → cluster → persist to DuckDB + LanceDB.

    Args:
        pdf_dir: Directory containing PDF files.
        db_path: DuckDB database path (default: data/clustering_v2.duckdb).
        lance_path: LanceDB directory path (default: data/vectors.lance).
        parsed_text_dir: Where to save extracted .txt files (default: data/parsed_texts/).
        device: Torch device for embeddings ("cuda" or "cpu").
        callback: Optional progress callback(stage, message).

    Returns:
        {
            "store": ClusteringStore,
            "lance_store": LanceVectorStore,
            "run_id": str,
            "stats": dict,
            "n_documents": int,
            "n_chunks": int,
            "n_domains": int,
            "n_clusters": int,
        }
    """
    def progress(stage: str, msg: str):
        logger.info(f"[{stage}] {msg}")
        if callback:
            callback(stage, msg)

    # ── Initialize stores ──────────────────────────────────────────────────
    db = db_path or DB_PATH
    lance = lance_path or LANCEDB_PATH
    store = ClusteringStore(db)
    lance_store = LanceVectorStore(lance)
    run_id = f"pdf_run_{uuid.uuid4().hex[:8]}"
    store.start_run(run_id, "init")

    progress("init", f"DuckDB={db}, LanceDB={lance}, cuML={cuml_available()}, device={device}")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1: Parse PDFs
    # ══════════════════════════════════════════════════════════════════════
    progress("parse", f"Parsing PDFs from {pdf_dir}...")
    parsed_docs = parse_pdf_directory(pdf_dir, parsed_text_dir)
    if not parsed_docs:
        progress("error", "No PDFs parsed successfully")
        return {"store": store, "lance_store": lance_store, "run_id": run_id,
                "stats": {}, "n_documents": 0, "n_chunks": 0, "n_domains": 0, "n_clusters": 0}

    # Build agreement dicts
    agreements = [
        {
            "agreement_id": doc["agreement_id"],
            "filename": doc["filename"],
            "raw_text": doc["raw_text"],
        }
        for doc in parsed_docs
    ]
    progress("parse", f"Parsed {len(agreements)} PDFs ({sum(d['char_count'] for d in parsed_docs):,} chars)")
    store.update_run(run_id, "parse_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2: Chunk documents
    # ══════════════════════════════════════════════════════════════════════
    # Initialize embedder early — needed for both chunking and embedding
    embedder = NomicEmbedder.get_instance(device=device)

    progress("chunk", "Semantic chunking with Nomic embeddings...")
    all_chunks: list[dict] = []
    chunker_config = SemanticChunkerConfig(
        buffer_size=1,
        breakpoint_percentile=90.0,
        min_chunk_chars=100,
        max_chunk_chars=2000,
        neighbor_weight=0.3,
        structural_bonus_strong=0.15,
        structural_bonus_weak=0.08,
    )

    for agr in agreements:
        chunks = semantic_chunk(
            agr["raw_text"],
            embed_fn=embedder.embed_for_clustering,
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

    progress("chunk", f"{len(all_chunks)} semantic chunks from {len(agreements)} documents")
    store.update_run(run_id, "chunk_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 3: Embed final chunks with Nomic on GPU
    # ══════════════════════════════════════════════════════════════════════
    progress("embed", f"Computing final chunk embeddings on {device}...")

    chunk_texts = [c["text"] for c in all_chunks]
    chunk_ids = [c["chunk_id"] for c in all_chunks]
    chunk_agr_ids = [c["agreement_id"] for c in all_chunks]

    # Re-embed final chunks (the chunker embedded sentence buffers internally,
    # but we need embeddings of the assembled chunks for clustering)
    chunk_embeddings = embedder.embed_for_clustering(chunk_texts, show_progress=True)

    # Field-aware summaries for macro clustering (legacy fields_processor approach)
    from core.macro_clusterer import get_field_summaries
    summaries = get_field_summaries(
        agreements, all_chunks, chunk_embeddings,
        embed_fn=embedder.embed_for_clustering,
    )
    summary_embeddings = embedder.embed_for_clustering(summaries)

    progress("embed", f"{len(all_chunks)} chunks + {len(agreements)} summaries ({embedder.dim}-dim)")

    # ── Persist chunks to DuckDB ───────────────────────────────────────────
    for i, chunk in enumerate(all_chunks):
        store.insert_chunk(
            chunk["chunk_id"], chunk["agreement_id"], chunk["text"],
            chunk["chunk_index"], chunk_embeddings[i], chunk.get("token_count", 0),
        )

    # ── Persist vectors to LanceDB ─────────────────────────────────────────
    lance_store.upsert_chunks(
        chunk_ids=chunk_ids,
        vectors=chunk_embeddings,
        texts=chunk_texts,
        agreement_ids=chunk_agr_ids,
    )
    lance_store.upsert_documents(
        agreement_ids=[a["agreement_id"] for a in agreements],
        vectors=summary_embeddings,
        filenames=[a.get("filename", "") for a in agreements],
        summaries=summaries,
    )

    progress("embed", f"Persisted to DuckDB + LanceDB")
    store.update_run(run_id, "embed_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 4: Macro clustering (document-level domains)
    # ══════════════════════════════════════════════════════════════════════
    progress("macro", "Clustering documents into domains...")
    n_docs = len(summaries)
    macro_umap = make_umap(
        n_neighbors=min(15, n_docs - 1),
        n_components=min(5, max(2, n_docs - 2)),
        min_dist=0.0,
        metric="cosine",
    )
    macro_hdbscan = make_hdbscan(
        min_cluster_size=MACRO_MIN_CLUSTER_SIZE,
        min_samples=MACRO_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    macro_vectorizer = CountVectorizer(
        stop_words=list(set(LEGAL_STOPWORDS)),
        ngram_range=(1, 2),
        min_df=max(1, min(2, n_docs // 5)),
    )
    _calc_probs = cuml_available()
    macro_model = BERTopic(
        umap_model=macro_umap,
        hdbscan_model=macro_hdbscan,
        vectorizer_model=macro_vectorizer,
        nr_topics="auto",
        calculate_probabilities=_calc_probs,
        verbose=False,
    )
    macro_topics, _ = macro_model.fit_transform(summaries, summary_embeddings)

    # Build domain records with auto-labels (no LLM)
    unique_macro = sorted(set(t for t in macro_topics if t != -1))
    domain_map: dict[int, dict] = {}  # topic_id → domain record

    for topic_id in unique_macro:
        mask = [i for i, t in enumerate(macro_topics) if t == topic_id]
        centroid = summary_embeddings[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        info = _auto_label_topic(macro_model, topic_id)
        domain_id = f"d_{uuid.uuid4().hex[:8]}"

        domain_map[topic_id] = {
            "domain_id": domain_id,
            "topic_id": topic_id,
            "label": info["label"],
            "description": info["description"],
            "agreement_count": len(mask),
            "centroid": centroid,
            "agreement_indices": mask,
        }

        store.upsert_domain(
            domain_id, info["label"], info["description"],
            len(mask), float(np.mean([np.dot(summary_embeddings[i], centroid) for i in mask])),
            centroid,
        )
        for idx in mask:
            agr = agreements[idx]
            conf = float(np.dot(summary_embeddings[idx], centroid))
            store.update_agreement_domain(agr["agreement_id"], domain_id, conf)

    # Handle outliers
    outlier_mask = [i for i, t in enumerate(macro_topics) if t == -1]
    if outlier_mask:
        outlier_domain_id = f"d_{uuid.uuid4().hex[:8]}"
        store.upsert_domain(outlier_domain_id, "Uncategorized", "Unassigned documents", len(outlier_mask))
        domain_map[-1] = {
            "domain_id": outlier_domain_id, "topic_id": -1,
            "label": "Uncategorized", "agreement_indices": outlier_mask,
        }
        for idx in outlier_mask:
            store.update_agreement_domain(agreements[idx]["agreement_id"], outlier_domain_id, 0.0)

    n_domains = len(unique_macro)
    progress("macro", f"{n_domains} domains from {n_docs} documents")
    store.update_run(run_id, "macro_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 5: Micro clustering per domain (clause-level)
    # ══════════════════════════════════════════════════════════════════════
    progress("micro", "Micro-clustering within each domain...")
    merger = HierarchicalTopicMerger()
    total_clusters = 0

    for topic_id, dr in domain_map.items():
        if topic_id == -1:
            continue
        domain_id = dr["domain_id"]
        domain_label = dr["label"]

        # Gather chunks belonging to this domain's agreements
        domain_agr_ids = set(agreements[i]["agreement_id"] for i in dr["agreement_indices"])
        domain_chunk_idx = [
            i for i, c in enumerate(all_chunks)
            if c["agreement_id"] in domain_agr_ids
        ]

        if len(domain_chunk_idx) < 5:
            progress("micro", f"  Skipping {domain_label} — {len(domain_chunk_idx)} chunks (< 5)")
            continue

        d_texts = [all_chunks[i]["text"] for i in domain_chunk_idx]
        d_ids = [all_chunks[i]["chunk_id"] for i in domain_chunk_idx]
        d_embs = chunk_embeddings[domain_chunk_idx]

        # cuML UMAP/HDBSCAN
        n = len(d_texts)
        n_neighbors = min(UMAP_N_NEIGHBORS, n - 1)
        n_components = max(2, min(UMAP_N_COMPONENTS, n - 2, n_neighbors - 1))

        micro_umap = make_umap(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=UMAP_MIN_DIST,
            metric="cosine",
        )
        micro_hdbscan = make_hdbscan(
            min_cluster_size=max(MICRO_MIN_CLUSTER_SIZE, 2),
            min_samples=max(MICRO_MIN_SAMPLES, 1),
            metric="euclidean",
            cluster_selection_method="eom",
        )
        micro_vectorizer = CountVectorizer(
            stop_words=list(set(LEGAL_STOPWORDS)),
            ngram_range=(1, 2),
            min_df=max(1, min(2, n // 10)),
            max_features=5000,
        )
        micro_model = BERTopic(
            umap_model=micro_umap,
            hdbscan_model=micro_hdbscan,
            vectorizer_model=micro_vectorizer,
            nr_topics="auto",
            calculate_probabilities=_calc_probs,
            verbose=False,
        )
        micro_topics, _ = micro_model.fit_transform(d_texts, d_embs)

        # Outlier reduction
        if -1 in micro_topics:
            try:
                new_topics = micro_model.reduce_outliers(
                    d_texts, micro_topics, strategy="c-tf-idf", threshold=0.3,
                )
                micro_model.update_topics(d_texts, topics=new_topics)
                micro_topics = new_topics
            except Exception:
                pass

        # Quality scores + build records
        quality = compute_cluster_quality(d_embs, micro_topics)
        unique_micro = sorted(set(t for t in micro_topics if t != -1))

        cluster_records = []
        for tid in unique_micro:
            mask = [i for i, t in enumerate(micro_topics) if t == tid]
            c_embs = d_embs[mask]
            centroid = c_embs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            keywords = [w for w, _ in micro_model.get_topic(tid)][:15]
            info = _auto_label_topic(micro_model, tid)
            cluster_id = f"c_{uuid.uuid4().hex[:8]}"

            cluster_records.append({
                "cluster_id": cluster_id,
                "topic_id": tid,
                "domain_id": domain_id,
                "label": info["label"],
                "description": info["description"],
                "keywords": keywords,
                "chunk_count": len(mask),
                "quality_score": quality.get(tid, 0.0),
                "centroid": centroid,
                "chunk_ids": [d_ids[i] for i in mask],
            })

        # Topic merging (FIELD-310)
        if len(cluster_records) > 1:
            cluster_records = merger.merge_similar_topics(cluster_records)

        # ── Persist clusters to DuckDB ─────────────────────────────────────
        for cr in cluster_records:
            c_chunk_ids = cr.get("chunk_ids", [])
            c_agr_ids = set()
            for cid in c_chunk_ids:
                for c in all_chunks:
                    if c["chunk_id"] == cid:
                        c_agr_ids.add(c["agreement_id"])
                        break

            store.upsert_cluster(
                cr["cluster_id"], domain_id, cr["label"], cr.get("description", ""),
                cr.get("keywords", []), cr["chunk_count"],
                len(c_agr_ids), cr.get("quality_score"),
                cr.get("centroid"), cr.get("aliases", []), cr.get("merged_from", []),
            )
            if cr.get("centroid") is not None:
                store.save_centroid(cr["cluster_id"], cr["centroid"], cr["chunk_count"])
            for cid in c_chunk_ids:
                store.insert_cluster_assignment(cid, cr["cluster_id"])

        # ── Update LanceDB with cluster assignments ────────────────────────
        lance_cids, lance_dids, lance_clids = [], [], []
        for cr in cluster_records:
            for cid in cr.get("chunk_ids", []):
                lance_cids.append(cid)
                lance_dids.append(domain_id)
                lance_clids.append(cr.get("topic_id", -1))
        if lance_cids:
            lance_store.update_cluster_assignments(lance_cids, lance_dids, lance_clids)

        n_micro = len(cluster_records)
        total_clusters += n_micro
        progress("micro", f"  {domain_label}: {n_micro} clause types from {n} chunks")

    store.update_run(run_id, "completed", "completed")
    stats = store.get_stats()
    progress("done", f"Pipeline complete: {len(agreements)} docs, {len(all_chunks)} chunks, "
             f"{n_domains} domains, {total_clusters} clusters")

    return {
        "store": store,
        "lance_store": lance_store,
        "run_id": run_id,
        "stats": stats,
        "n_documents": len(agreements),
        "n_chunks": len(all_chunks),
        "n_domains": n_domains,
        "n_clusters": total_clusters,
    }


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m core.pdf_pipeline <pdf_dir> [--db <path>] [--lance <path>]")
        sys.exit(1)

    pdf_dir = sys.argv[1]
    db = None
    lance = None
    for i, arg in enumerate(sys.argv):
        if arg == "--db" and i + 1 < len(sys.argv):
            db = sys.argv[i + 1]
        if arg == "--lance" and i + 1 < len(sys.argv):
            lance = sys.argv[i + 1]

    result = run_pdf_pipeline(pdf_dir, db_path=db, lance_path=lance)
    print(f"\nDone: {result['n_documents']} docs → {result['n_chunks']} chunks → "
          f"{result['n_domains']} domains → {result['n_clusters']} clusters")
