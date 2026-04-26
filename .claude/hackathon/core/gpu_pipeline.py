"""
GPU-accelerated pipeline orchestrator.

Upgrades the CPU pipeline.py to use:
  - NomicEmbedder (GPU, 768-dim) instead of MiniLM (CPU, 384-dim)
  - cuML UMAP/HDBSCAN (GPU) instead of sklearn (CPU)
  - LanceDB for persistent vector storage alongside DuckDB metadata
  - Ray Tune Bayesian optimization for hyperparameter search (50 trials)

Same 7-stage flow as the CPU pipeline, but each stage is GPU-accelerated.
Falls back gracefully to CPU when CUDA/cuML/Ray are unavailable.
"""
import json
import logging
import os
import uuid
from pathlib import Path

import numpy as np

from config import (
    DB_PATH, SAMPLE_DIR, LEGAL_STOPWORDS,
    MACRO_MIN_CLUSTER_SIZE, MACRO_MIN_SAMPLES,
    MICRO_MIN_CLUSTER_SIZE, MICRO_MIN_SAMPLES,
)
from core.store import ClusteringStore
from core.nomic_embedder import NomicEmbedder
from core.lancedb_store import LanceVectorStore
from core.cuml_wrappers import make_umap, make_hdbscan, cuml_available
from core.scoring import calculate_scores, apply_penalties
from core.llm_client import LLMClient
from core.chunker import semantic_chunk, get_document_summary, SemanticChunkerConfig
from core.topic_merger import HierarchicalTopicMerger
from core.field_discovery import discover_fields
from core.extractor import extract_conditioned

logger = logging.getLogger(__name__)

# ── GPU config (from env or defaults) ──────────────────────────────────────────
NOMIC_MODEL = os.getenv("NOMIC_MODEL", "nomic-ai/nomic-embed-text-v1.5")
NOMIC_DEVICE = os.getenv("NOMIC_DEVICE", "cuda")
NOMIC_DIM = int(os.getenv("NOMIC_DIM", "768"))  # Full dim; set 256/512 for Matryoshka
LANCEDB_PATH = os.getenv("LANCEDB_PATH", str(Path(DB_PATH).parent / "vectors.lance"))
OPTIMIZE_TRIALS = int(os.getenv("OPTIMIZE_TRIALS", "50"))
OPTIMIZE_PARALLEL = int(os.getenv("OPTIMIZE_PARALLEL", "4"))
USE_RAY = os.getenv("USE_RAY", "true").lower() in ("1", "true", "yes")


def _load_documents(data_dir: Path) -> list[dict]:
    """Load sample agreements from text/json/pdf files."""
    agreements = []
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

    for fpath in sorted(data_dir.glob("*.txt")):
        agreements.append({
            "agreement_id": f"agr_{uuid.uuid4().hex[:8]}",
            "filename": fpath.name,
            "raw_text": fpath.read_text(errors="replace"),
        })
    return agreements


def run_gpu_pipeline(
    agreements: list[dict] | None = None,
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    lance_path: str | Path | None = None,
    skip_extraction: bool = False,
    skip_optimization: bool = False,
    n_trials: int | None = None,
    callback=None,
) -> dict:
    """
    Run the full GPU-accelerated V2 clustering pipeline.

    Stages:
      1. Load documents
      2. Chunk documents
      3. Embed with Nomic on GPU + store in LanceDB
      4. Macro cluster (doc-level BERTopic with cuML)
      5. Micro cluster per domain (clause-level, optional Bayesian optimization)
      6. Field discovery (LLM)
      7. Field extraction (LLM)

    Args:
        agreements: Pre-loaded agreements (optional).
        data_dir: Path to sample data directory.
        db_path: Path to DuckDB database.
        lance_path: Path to LanceDB directory.
        skip_extraction: Skip LLM field discovery/extraction.
        skip_optimization: Skip Bayesian hyperparameter optimization.
        n_trials: Override number of optimization trials.
        callback: Progress callback(stage, message).

    Returns:
        Dict with store, lance_store, run_id, and stats.
    """
    def progress(stage: str, msg: str):
        logger.info(f"[GPU-{stage}] {msg}")
        if callback:
            callback(stage, msg)

    # ── Initialize ─────────────────────────────────────────────────────────
    store = ClusteringStore(db_path or DB_PATH)
    lance_store = LanceVectorStore(lance_path or LANCEDB_PATH)
    embedder = NomicEmbedder.get_instance(
        model_name=NOMIC_MODEL,
        device=NOMIC_DEVICE,
        matryoshka_dim=NOMIC_DIM if NOMIC_DIM < 768 else None,
    )
    llm = LLMClient()
    run_id = f"gpu_run_{uuid.uuid4().hex[:8]}"
    store.start_run(run_id, "init")

    progress("init", f"cuML={cuml_available()}, embedder={NOMIC_MODEL}, dim={embedder.dim}")

    # ── Initialize Ray early (if optimization will be used) ────────────────
    if USE_RAY and not skip_optimization:
        from core.ray_optimizer import init_ray
        cluster_info = init_ray()
        progress("init", f"Ray: {cluster_info['cpus']} CPUs, {cluster_info['gpus']} GPUs")

    # ── Stage 1: Load documents ────────────────────────────────────────────
    progress("load", "Loading documents...")
    if agreements is None:
        agreements = _load_documents(Path(data_dir) if data_dir else SAMPLE_DIR)
    if not agreements:
        progress("error", "No documents found!")
        return {"store": store, "lance_store": lance_store, "run_id": run_id, "stats": {}}
    progress("load", f"Loaded {len(agreements)} agreements")

    # ── Stage 2: Semantic chunk documents ──────────────────────────────────
    progress("chunk", "Semantic chunking with Nomic embeddings...")
    chunker_config = SemanticChunkerConfig(
        buffer_size=1,
        breakpoint_percentile=90.0,
        min_chunk_chars=100,
        max_chunk_chars=2000,
        neighbor_weight=0.3,
        structural_bonus_strong=0.15,
        structural_bonus_weak=0.08,
    )
    all_chunks = []
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
    progress("chunk", f"Created {len(all_chunks)} semantic chunks from {len(agreements)} documents")

    # ── Stage 3: Embed final chunks with Nomic on GPU ──────────────────────
    progress("embed", f"Computing final chunk embeddings with {NOMIC_MODEL} on {NOMIC_DEVICE}...")
    chunk_texts = [c["text"] for c in all_chunks]
    chunk_ids = [c["chunk_id"] for c in all_chunks]
    agreement_ids_for_chunks = [c["agreement_id"] for c in all_chunks]

    chunk_embeddings = embedder.embed_for_clustering(chunk_texts, show_progress=True)
    progress("embed", f"Embedded {len(all_chunks)} clauses ({embedder.dim}-dim on GPU)")

    # Store chunks in DuckDB
    for i, chunk in enumerate(all_chunks):
        store.insert_chunk(
            chunk["chunk_id"], chunk["agreement_id"], chunk["text"],
            chunk["chunk_index"], chunk_embeddings[i], chunk.get("token_count", 0),
        )

    # Store vectors in LanceDB
    lance_store.upsert_chunks(
        chunk_ids=chunk_ids,
        vectors=chunk_embeddings,
        texts=chunk_texts,
        agreement_ids=agreement_ids_for_chunks,
    )
    progress("embed", f"Stored {len(all_chunks)} vectors in LanceDB")

    # ── Stage 4: Macro clustering with cuML ────────────────────────────────
    progress("macro", "Building field-aware summaries for macro clustering...")
    from core.macro_clusterer import get_field_summaries, label_macro_clusters, build_domain_records

    summaries = get_field_summaries(
        agreements, all_chunks, chunk_embeddings,
        embed_fn=embedder.embed_for_clustering,
    )

    summary_embeddings = embedder.embed_for_clustering(summaries)

    # Store document vectors in LanceDB
    lance_store.upsert_documents(
        agreement_ids=[a["agreement_id"] for a in agreements],
        vectors=summary_embeddings,
        filenames=[a.get("filename", "") for a in agreements],
        summaries=summaries,
    )

    # BERTopic with cuML wrappers
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    n_docs = len(summaries)
    umap_model = make_umap(
        n_neighbors=min(15, n_docs - 1),
        n_components=min(5, n_docs - 2),
        min_dist=0.0,
        metric="cosine",
    )
    hdbscan_model = make_hdbscan(
        min_cluster_size=MACRO_MIN_CLUSTER_SIZE,
        min_samples=MACRO_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    vectorizer = CountVectorizer(
        stop_words=list(set(LEGAL_STOPWORDS)),
        ngram_range=(1, 2),
        min_df=2,
    )
    macro_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=False,
    )
    topics, _probs = macro_model.fit_transform(summaries, summary_embeddings)
    n_domains = len(set(t for t in topics if t != -1))
    progress("macro", f"Found {n_domains} macro domains")

    # Label and store macro domains
    from core.macro_clusterer import label_macro_clusters, build_domain_records
    macro_labels = label_macro_clusters(llm, macro_model)
    domain_records = build_domain_records(agreements, topics, summary_embeddings, macro_labels)

    topic_to_domain = {}
    for dr in domain_records:
        store.upsert_domain(
            dr["domain_id"], dr["label"], dr["description"],
            dr["agreement_count"], dr["confidence"], dr.get("centroid"),
        )
        topic_to_domain[dr["topic_id"]] = dr["domain_id"]
        for idx in dr["agreement_indices"]:
            agr = agreements[idx]
            conf = float(np.dot(summary_embeddings[idx], dr["centroid"])) if dr.get("centroid") is not None else 0.0
            store.update_agreement_domain(agr["agreement_id"], dr["domain_id"], conf)

    store.update_run(run_id, "macro_complete")

    # ── Stage 5: Micro clustering per domain (with optional optimization) ──
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
            progress("micro", f"  Skipping {domain_label} — too few chunks")
            continue

        domain_chunk_texts = [all_chunks[i]["text"] for i in domain_chunk_indices]
        domain_chunk_ids = [all_chunks[i]["chunk_id"] for i in domain_chunk_indices]
        domain_embeddings = chunk_embeddings[domain_chunk_indices]

        # ── Bayesian optimization (optional) ──────────────────────────
        if not skip_optimization and len(domain_chunk_texts) >= 20:
            progress("optimize", f"  Optimizing {domain_label} ({len(domain_chunk_texts)} chunks)...")
            trials = n_trials or OPTIMIZE_TRIALS

            # Scale thresholds to domain size
            domain_min_clusters = max(3, len(domain_chunk_texts) // 10)

            if USE_RAY:
                try:
                    from core.ray_optimizer import optimize_with_ray
                    opt_result = optimize_with_ray(
                        docs=domain_chunk_texts,
                        embeddings=domain_embeddings,
                        n_trials=trials,
                        n_parallel=OPTIMIZE_PARALLEL,
                        stopwords=list(set(LEGAL_STOPWORDS)),
                        min_clusters=domain_min_clusters,
                    )
                    best_params = opt_result.best_params
                    progress("optimize", f"  Ray: best_score={opt_result.best_score:.4f} in {opt_result.elapsed_seconds:.1f}s")
                except ImportError:
                    logger.warning("Ray not available, falling back to sequential")
                    from core.ray_optimizer import optimize_sequential
                    opt_result = optimize_sequential(
                        docs=domain_chunk_texts,
                        embeddings=domain_embeddings,
                        n_trials=min(trials, 30),
                        stopwords=list(set(LEGAL_STOPWORDS)),
                        min_clusters=domain_min_clusters,
                    )
                    best_params = opt_result.best_params
            else:
                from core.ray_optimizer import optimize_sequential
                opt_result = optimize_sequential(
                    docs=domain_chunk_texts,
                    embeddings=domain_embeddings,
                    n_trials=min(trials, 30),
                    stopwords=list(set(LEGAL_STOPWORDS)),
                    min_clusters=domain_min_clusters,
                )
                best_params = opt_result.best_params

            # Build final model with best params
            n = len(domain_chunk_texts)
            umap_model = make_umap(
                n_neighbors=min(int(best_params["umap_n_neighbors"]), n - 1),
                n_components=max(2, min(int(best_params["umap_n_components"]), n - 2)),
                min_dist=float(best_params["umap_min_dist"]),
                metric="cosine",
            )
            hdbscan_model = make_hdbscan(
                min_cluster_size=max(2, int(best_params["hdbscan_min_cluster_size"])),
                min_samples=max(1, int(best_params["hdbscan_min_samples"])),
                metric="euclidean",
                cluster_selection_method="eom",
            )
        else:
            # Default params (same as CPU pipeline)
            umap_model = make_umap(
                n_neighbors=min(15, len(domain_chunk_texts) - 1),
                n_components=max(2, min(5, len(domain_chunk_texts) - 2)),
                min_dist=0.0,
                metric="cosine",
            )
            hdbscan_model = make_hdbscan(
                min_cluster_size=max(MICRO_MIN_CLUSTER_SIZE, 2),
                min_samples=max(MICRO_MIN_SAMPLES, 1),
                metric="euclidean",
                cluster_selection_method="eom",
            )

        # Build and fit BERTopic
        vectorizer = CountVectorizer(
            stop_words=list(set(LEGAL_STOPWORDS)),
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000,
        )
        micro_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            nr_topics="auto",
            calculate_probabilities=True,
            verbose=False,
        )
        micro_topics, _ = micro_model.fit_transform(domain_chunk_texts, domain_embeddings)

        # Outlier reduction
        if -1 in micro_topics:
            try:
                new_topics = micro_model.reduce_outliers(
                    domain_chunk_texts, micro_topics,
                    strategy="c-tf-idf", threshold=0.3,
                )
                micro_model.update_topics(domain_chunk_texts, topics=new_topics)
                micro_topics = new_topics
            except Exception:
                pass

        # Compute quality scores
        from core.micro_clusterer import compute_cluster_quality, label_micro_clusters, build_cluster_records
        quality = compute_cluster_quality(domain_embeddings, micro_topics)
        micro_labels = label_micro_clusters(llm, micro_model, domain_label)
        cluster_records = build_cluster_records(
            domain_id, domain_chunk_ids, micro_topics,
            domain_embeddings, micro_labels, quality, micro_model,
        )

        # Topic merging (FIELD-310)
        if len(cluster_records) > 1:
            cluster_records = merger.merge_similar_topics(cluster_records)

        # Store clusters
        for cr in cluster_records:
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
            if cr.get("centroid") is not None:
                store.save_centroid(cr["cluster_id"], cr["centroid"], cr["chunk_count"])
            for cid in cluster_chunk_ids:
                store.insert_cluster_assignment(cid, cr["cluster_id"])

        # Update LanceDB with cluster assignments
        domain_cids = []
        domain_dids = []
        domain_clids = []
        for cr in cluster_records:
            for cid in cr.get("chunk_ids", []):
                domain_cids.append(cid)
                domain_dids.append(domain_id)
                domain_clids.append(cr.get("topic_id", -1))
        if domain_cids:
            lance_store.update_cluster_assignments(domain_cids, domain_dids, domain_clids)

        all_cluster_records.extend(cluster_records)

    store.update_run(run_id, "micro_complete")
    n_clusters = len(all_cluster_records)
    progress("micro", f"Total: {n_clusters} clause types across all domains")

    # ── Stage 6: Field discovery ───────────────────────────────────────────
    if not skip_extraction:
        progress("fields", "Discovering fields per cluster...")
        for cr in all_cluster_records:
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

    # ── Stage 7: Field extraction ──────────────────────────────────────────
    if not skip_extraction:
        progress("extract", "Extracting fields from example sets...")
        for cr in all_cluster_records:
            fields = store.get_fields(cr["cluster_id"])
            if not fields:
                continue
            rep_chunk_ids = cr.get("chunk_ids", [])[:20]
            for cid in rep_chunk_ids:
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
    progress("done", f"GPU pipeline complete: {stats}")

    return {
        "store": store,
        "lance_store": lance_store,
        "run_id": run_id,
        "stats": stats,
    }
