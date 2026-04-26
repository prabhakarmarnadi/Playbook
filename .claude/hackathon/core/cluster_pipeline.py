"""
Production featurizer + clustering pipeline with model persistence.

Runs the full: parse → chunk → embed → macro cluster → micro cluster flow
with BERTopic model weights saved locally for later incremental assignment.

Artifacts stored per account_id under data/artifacts/<account_id>/:
  - macro_model/          BERTopic safetensors + config
  - micro_models/<domain_id>/   BERTopic per-domain models
  - embeddings.npz        chunk + summary embeddings
  - run_manifest.json     run metadata + stats

Usage:
    # CLI
    poetry run python -m core.cluster_pipeline /path/to/pdfs --account-id acct_001

    # Python
    from core.cluster_pipeline import run_cluster_pipeline
    result = run_cluster_pipeline("/path/to/pdfs", account_id="acct_001")
"""
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from config import (
    DB_PATH, DATA_DIR, LEGAL_STOPWORDS,
    MACRO_MIN_CLUSTER_SIZE, MACRO_MIN_SAMPLES,
    MICRO_MIN_CLUSTER_SIZE, MICRO_MIN_SAMPLES,
    UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS, UMAP_MIN_DIST,
    OPTIMIZE_TRIALS,
)
from core.pdf_parser import parse_pdf_directory, load_parsed_agreements, PARSED_TEXT_DIR
from core.chunker import semantic_chunk, sliding_window_chunk, get_document_summary, SemanticChunkerConfig
from core.nomic_embedder import NomicEmbedder
from core.cuml_wrappers import make_umap, make_hdbscan, cuml_available
from core.lancedb_store import LanceVectorStore
from core.store import ClusteringStore
from core.topic_merger import HierarchicalTopicMerger
from core.micro_clusterer import compute_cluster_quality
from core.nupunkt_chunker import nupunkt_available
from core.chunker import structural_chunk
from core.keybert_scorer import KeyBERTScorer, keybert_available
from config import KEYBERT_UMAP_TARGET_WEIGHT, KEYBERT_PRIOR_WEIGHT, ENABLE_ARM_ENRICHMENT

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = DATA_DIR / "artifacts"

# Combined stopword set (English + legal domain) — matches legacy
COMBINED_STOPWORDS = set(ENGLISH_STOP_WORDS) | set(LEGAL_STOPWORDS)


class StopwordFilteringAnalyzer:
    """Picklable custom analyzer that filters n-grams where ALL tokens are stopwords.

    Ported from legacy bertopic_clusterer.py — ensures n-grams like
    'pursuant to' or 'in accordance with' are filtered out while keeping
    meaningful phrases like 'termination clause'.
    """

    def __init__(self, ngram_range: tuple = (1, 4), stopwords: set | None = None):
        self.ngram_range = ngram_range
        self.stopwords = stopwords or COMBINED_STOPWORDS
        self._analyzer = None

    def _build_analyzer(self):
        if self._analyzer is None:
            base_vectorizer = CountVectorizer(ngram_range=self.ngram_range)
            self._analyzer = base_vectorizer.build_analyzer()
        return self._analyzer

    def __call__(self, doc: str) -> list:
        analyzer = self._build_analyzer()
        tokens = analyzer(doc)
        filtered = []
        for token in tokens:
            words = token.split()
            if any(word not in self.stopwords for word in words):
                filtered.append(token)
        return filtered

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_analyzer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def _auto_label_topic(topic_model: BERTopic, topic_id: int, prefix: str = "") -> dict:
    if topic_id == -1:
        return {"label": "Uncategorized", "description": "Not assigned to any cluster"}
    keywords = [w for w, _ in topic_model.get_topic(topic_id)]
    top = keywords[:5]
    label = f"{prefix}{' / '.join(top[:3]).title()}" if top else f"{prefix}Topic_{topic_id}"
    desc = f"Keywords: {', '.join(top)}" if top else ""
    return {"label": label, "description": desc}


def _save_model(model: BERTopic, path: Path, name: str = "model"):
    """Save BERTopic model + embedding_model=None (we embed separately)."""
    path.mkdir(parents=True, exist_ok=True)
    save_path = path / name
    try:
        model.save(
            str(save_path),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=False,
        )
        logger.info(f"Saved BERTopic model: {save_path}")
    except (TypeError, Exception) as e:
        # BERTopic save can fail with numpy int64 JSON serialization;
        # fall back to pickle serialization in a separate path
        import shutil
        if save_path.exists():
            shutil.rmtree(save_path, ignore_errors=True)
        model.save(
            str(save_path),
            serialization="pickle",
            save_ctfidf=True,
            save_embedding_model=False,
        )
        logger.info(f"Saved BERTopic model (pickle fallback): {save_path}")


def _save_embeddings(path: Path, chunk_embeddings: np.ndarray, summary_embeddings: np.ndarray):
    """Save embeddings as compressed npz for re-use."""
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "embeddings.npz",
        chunk_embeddings=chunk_embeddings,
        summary_embeddings=summary_embeddings,
    )
    logger.info(f"Saved embeddings: {path / 'embeddings.npz'} "
                f"(chunks={chunk_embeddings.shape}, summaries={summary_embeddings.shape})")


def _load_cached_embeddings(account_dir: Path, expected_chunks: int, expected_summaries: int):
    """Try to load cached embeddings from the most recent run for this account.
    
    Returns (chunk_embeddings, summary_embeddings) if cache hit, else (None, None).
    """
    if not account_dir.exists():
        return None, None
    # Find most recent run dir with embeddings
    run_dirs = sorted(
        [d for d in account_dir.iterdir() if d.is_dir() and (d / "embeddings.npz").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        return None, None
    cache_path = run_dirs[0] / "embeddings.npz"
    try:
        data = np.load(cache_path)
        chunk_emb = data["chunk_embeddings"]
        summary_emb = data["summary_embeddings"]
        if chunk_emb.shape[0] == expected_chunks and summary_emb.shape[0] == expected_summaries:
            logger.info(f"Embedding cache HIT: {cache_path} "
                        f"(chunks={chunk_emb.shape}, summaries={summary_emb.shape})")
            return chunk_emb, summary_emb
        logger.info(f"Embedding cache MISS (shape mismatch): cached chunks={chunk_emb.shape[0]} "
                     f"vs expected={expected_chunks}, cached summaries={summary_emb.shape[0]} "
                     f"vs expected={expected_summaries}")
    except Exception as e:
        logger.warning(f"Failed to load cached embeddings from {cache_path}: {e}")
    return None, None


def _save_manifest(path: Path, manifest: dict):
    """Save run manifest JSON."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Saved manifest: {path / 'run_manifest.json'}")


def _save_quality_report(path: Path, report: dict):
    """Save cluster quality evaluation report."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "cluster_quality.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved quality report: {path / 'cluster_quality.json'}")


def run_cluster_pipeline(
    pdf_dir: str | Path,
    account_id: str = "default",
    db_path: str | Path | None = None,
    lance_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    device: str = "cuda",
    max_docs: int | None = None,
    use_semantic_chunking: bool = True,
    use_rlm: bool = False,
    callback=None,
) -> dict:
    """
    Parse PDFs → chunk → embed → macro cluster → micro cluster → save artifacts.

    All BERTopic models and embeddings are persisted locally under
    <artifacts_dir>/<account_id>/<run_id>/ for later incremental assignment.

    Args:
        pdf_dir: Directory containing PDF files (searched recursively).
        account_id: Tenant ID for artifact isolation.
        db_path: DuckDB path (default: data/<account_id>.duckdb).
        lance_path: LanceDB path (default: data/<account_id>.lance).
        artifacts_dir: Root for model artifacts (default: data/artifacts/).
        device: "cuda" or "cpu".
        max_docs: Limit number of documents (for testing).
        use_semantic_chunking: If False, use fast sliding window (for large batches).
        callback: Progress callback(stage, message).

    Returns:
        Dict with store, lance_store, run_id, stats, artifact paths.
    """
    t0 = time.time()
    stage_timings: dict[str, float] = {}  # stage_name -> seconds
    _stage_t0: dict[str, float] = {}  # for tracking in-progress stages

    def progress(stage: str, msg: str):
        elapsed = time.time() - t0
        logger.info(f"[{elapsed:6.1f}s] [{stage}] {msg}")
        if callback:
            callback(stage, msg)

    def stage_start(name: str):
        _stage_t0[name] = time.time()

    def stage_end(name: str):
        if name in _stage_t0:
            stage_timings[name] = round(time.time() - _stage_t0.pop(name), 2)
            progress(name, f"completed in {stage_timings[name]:.1f}s")

    # ── Paths ──────────────────────────────────────────────────────────────
    artifacts_root = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    account_dir = artifacts_root / account_id

    db = Path(db_path) if db_path else DATA_DIR / f"{account_id}.duckdb"
    lance = Path(lance_path) if lance_path else DATA_DIR / f"{account_id}.lance"

    store = ClusteringStore(db)
    lance_store = LanceVectorStore(lance)
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    run_dir = account_dir / run_id

    store.start_run(run_id, "init")
    progress("init", f"account={account_id} cuML={cuml_available()} device={device}")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1: Parse PDFs (with caching)
    # ══════════════════════════════════════════════════════════════════════
    stage_start("parse")

    # Check for previously parsed results first
    cached = load_parsed_agreements()
    pdf_dir_resolved = str(Path(pdf_dir).resolve())

    # Filter cached results to those from the same source directory
    manifest_path = PARSED_TEXT_DIR / "manifest.json"
    cache_hit = False
    if cached and manifest_path.exists():
        import json as _json
        with open(manifest_path) as _f:
            _manifest = _json.load(_f)
        if str(Path(_manifest.get("source_dir", "")).resolve()) == pdf_dir_resolved:
            cache_hit = True

    if cache_hit:
        progress("parse", f"Using {len(cached)} cached parsed PDFs from {PARSED_TEXT_DIR}")
        parsed_docs = cached
    else:
        progress("parse", f"Parsing PDFs from {pdf_dir} (no cache found)...")
        parsed_docs = parse_pdf_directory(pdf_dir)

    if max_docs and len(parsed_docs) > max_docs:
        parsed_docs = parsed_docs[:max_docs]
        progress("parse", f"Limited to {max_docs} documents")

    if not parsed_docs:
        progress("error", "No PDFs parsed")
        return {"store": store, "lance_store": lance_store, "run_id": run_id,
                "stats": {}, "n_documents": 0}

    agreements = [
        {"agreement_id": d["agreement_id"], "filename": d["filename"], "raw_text": d["raw_text"]}
        for d in parsed_docs
        if len(d.get("raw_text", "")) > 100  # skip near-empty PDFs
    ]
    total_chars = sum(len(a["raw_text"]) for a in agreements)
    progress("parse", f"{len(agreements)} valid PDFs ({total_chars:,} chars)")
    stage_end("parse")
    store.update_run(run_id, "parse_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2: Chunk
    # ══════════════════════════════════════════════════════════════════════
    stage_start("chunk")
    embedder = NomicEmbedder.get_instance(device=device)
    embedder.warmup()  # Pre-warm CUDA kernels on first run

    if use_semantic_chunking:
        progress("chunk", "Semantic chunking (clause-aware)...")
        chunker_config = SemanticChunkerConfig(
            buffer_size=1, breakpoint_percentile=90.0,
            min_chunk_chars=100, max_chunk_chars=2000,
            clause_split=True,
        )
        MAX_CHUNKS_PER_DOC = 80  # Truncate to avoid embedding explosion on huge docs

        from core.chunker import _structural_clause_split, _split_sentences, _combine_sentence_buffers, _compute_breakpoints, _assemble_chunks, _approx_token_count
        import uuid as _uuid

        # ── Phase 1: structural clause split (CPU, fast) ───────────────────
        # For each doc, split into structural sections (clauses).
        # Sections that fit within max_chunk_chars become chunks directly.
        # Oversized sections need semantic sub-splitting (requires embeddings).
        doc_clauses = []        # list of (agr, clauses: list[str])
        oversized_segments = [] # (doc_idx, clause_idx, text) for segments needing semantic split
        os_sentence_data = []   # (sentences, buffers) per oversized segment
        all_buffers = []        # flat buffer list for single batched embed
        buffer_offsets = []     # (start, end) per oversized segment into all_buffers

        for agr in agreements:
            raw_text = agr["raw_text"]
            sections = _structural_clause_split(raw_text, min_clause_chars=chunker_config.min_chunk_chars)
            if not sections:
                sections = [raw_text.strip()] if raw_text.strip() else []
            doc_clauses.append((agr, sections))

            for ci, section in enumerate(sections):
                if len(section) > chunker_config.max_chunk_chars:
                    # Needs semantic sub-splitting — prepare buffers
                    sentences = _split_sentences(section)
                    if sentences and len(sentences) > 2:
                        buffers = _combine_sentence_buffers(sentences, chunker_config.buffer_size)
                        os_idx = len(oversized_segments)
                        oversized_segments.append((len(doc_clauses) - 1, ci, section))
                        os_sentence_data.append((sentences, buffers))
                        buf_start = len(all_buffers)
                        all_buffers.extend(buffers)
                        buffer_offsets.append((buf_start, len(all_buffers)))
                    else:
                        # Too few sentences to sub-split, will be force-split later
                        oversized_segments.append((len(doc_clauses) - 1, ci, section))
                        os_sentence_data.append((sentences or [], []))
                        buffer_offsets.append((len(all_buffers), len(all_buffers)))

        n_direct = sum(
            sum(1 for s in sections if len(s) <= chunker_config.max_chunk_chars)
            for _, sections in doc_clauses
        )
        progress("chunk", f"Structural split: {sum(len(s) for _, s in doc_clauses)} sections "
                 f"({n_direct} direct, {len(oversized_segments)} need sub-splitting, "
                 f"{len(all_buffers)} buffers to embed)")

        # ── Phase 2: single batched embed for oversized segments ───────────
        # Truncate buffers to 512 chars — only need rough similarity for breakpoints
        truncated_buffers = [b[:512] for b in all_buffers]
        if truncated_buffers:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            all_buffer_embeddings = embedder.embed_for_clustering(
                truncated_buffers, show_progress=True, batch_size=8,
            )
        else:
            all_buffer_embeddings = np.empty((0, embedder.dim), dtype=np.float32)

        # Pre-compute semantic sub-chunks for oversized segments
        os_sub_chunks: dict[tuple[int, int], list[str]] = {}  # (doc_idx, clause_idx) -> chunk_texts
        for os_i, (doc_idx, clause_idx, section_text) in enumerate(oversized_segments):
            sentences, buffers = os_sentence_data[os_i]
            buf_start, buf_end = buffer_offsets[os_i]

            if buf_start < buf_end and sentences and len(sentences) > 2:
                doc_embeddings = all_buffer_embeddings[buf_start:buf_end]
                # Use a sub-config without clause_split to avoid re-splitting
                sub_cfg = SemanticChunkerConfig(
                    buffer_size=chunker_config.buffer_size,
                    breakpoint_percentile=chunker_config.breakpoint_percentile,
                    min_chunk_chars=chunker_config.min_chunk_chars,
                    max_chunk_chars=chunker_config.max_chunk_chars,
                    neighbor_weight=chunker_config.neighbor_weight,
                )
                breakpoints = _compute_breakpoints(doc_embeddings, sentences, sub_cfg)
                sub_texts = _assemble_chunks(sentences, breakpoints, sub_cfg)
            else:
                # Force-split by character limit
                from core.chunker import _force_split
                sub_texts = _force_split(sentences, chunker_config.max_chunk_chars) if sentences else [section_text]

            os_sub_chunks[(doc_idx, clause_idx)] = sub_texts

        # ── Phase 3: assemble final chunks per doc ─────────────────────────
        all_chunks = []
        for doc_idx, (agr, sections) in enumerate(doc_clauses):
            if (doc_idx + 1) % 200 == 0:
                progress("chunk", f"  {doc_idx+1}/{len(agreements)} documents assembled")
            chunk_idx = 0
            for ci, section in enumerate(sections):
                if (doc_idx, ci) in os_sub_chunks:
                    # Oversized: use pre-computed sub-chunks
                    sub_texts = os_sub_chunks[(doc_idx, ci)]
                else:
                    # Fits directly as one chunk
                    sub_texts = [section]

                for ct in sub_texts:
                    if chunk_idx >= MAX_CHUNKS_PER_DOC:
                        break
                    all_chunks.append({
                        "chunk_id": f"chk_{_uuid.uuid4().hex[:10]}",
                        "text": ct, "char_start": 0, "char_end": len(ct),
                        "token_count": _approx_token_count(ct), "chunk_index": chunk_idx,
                        "agreement_id": agr["agreement_id"],
                    })
                    chunk_idx += 1
    else:
        if nupunkt_available():
            progress("chunk", "Structural chunking (nupunkt + sliding window)...")
            all_chunks = []
            for agr in agreements:
                chunks = structural_chunk(agr["raw_text"])
                for c in chunks:
                    c["agreement_id"] = agr["agreement_id"]
                all_chunks.extend(chunks)
        else:
            progress("chunk", "Sliding window chunking (fast mode)...")
            all_chunks = []
            for agr in agreements:
                chunks = sliding_window_chunk(agr["raw_text"])
                for c in chunks:
                    c["agreement_id"] = agr["agreement_id"]
                all_chunks.extend(chunks)

    progress("chunk", f"{len(all_chunks)} chunks from {len(agreements)} docs")
    stage_end("chunk")

    # Store agreements in DuckDB
    for agr in agreements:
        summary = get_document_summary(agr["raw_text"], agr.get("filename", ""))
        store.insert_agreement(agr["agreement_id"], agr.get("filename", ""), agr["raw_text"], summary)

    store.update_run(run_id, "chunk_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 3: Embed
    # ══════════════════════════════════════════════════════════════════════
    stage_start("embed")
    chunk_texts = [c["text"] for c in all_chunks]

    # ── KeyBERT text augmentation ──
    keybert_scorer = None
    if keybert_available():
        try:
            keybert_scorer = KeyBERTScorer.get_instance()
            progress("embed", f"Augmenting {len(chunk_texts)} chunks with KeyBERT clause-type context...")
            chunk_texts = keybert_scorer.batch_augment(chunk_texts)
            progress("embed", f"KeyBERT augmentation applied ({len(keybert_scorer.clause_types)} clause types)")
        except Exception as e:
            logger.warning(f"KeyBERT augmentation failed, using raw text: {e}")
            keybert_scorer = None

    chunk_ids = [c["chunk_id"] for c in all_chunks]
    chunk_agr_ids = [c["agreement_id"] for c in all_chunks]

    # Build summaries first (cheap) so we can check cache with correct counts
    from core.macro_clusterer import get_field_summaries

    # Try loading cached embeddings from a previous run
    cached_chunk_emb, cached_summary_emb = _load_cached_embeddings(
        account_dir, expected_chunks=len(chunk_texts), expected_summaries=len(agreements),
    )

    if cached_chunk_emb is not None and cached_summary_emb is not None:
        progress("embed", f"Using cached embeddings ({cached_chunk_emb.shape[0]} chunks, {cached_summary_emb.shape[0]} summaries)")
        chunk_embeddings = cached_chunk_emb
        summary_embeddings = cached_summary_emb
        # Still need summaries for downstream stages
        summaries = get_field_summaries(
            agreements, all_chunks, chunk_embeddings,
            embed_fn=lambda texts, **kw: embedder.embed_for_clustering(texts, batch_size=8, **kw),
        )
    else:
        progress("embed", f"Embedding {len(all_chunks)} chunks on {device}...")
        chunk_embeddings = embedder.embed_for_clustering(chunk_texts, show_progress=True, batch_size=8)

        # Free VRAM before summary embedding
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Field-aware summaries
        progress("embed", "Building field-aware summaries...")
        summaries = get_field_summaries(
            agreements, all_chunks, chunk_embeddings,
            embed_fn=lambda texts, **kw: embedder.embed_for_clustering(texts, batch_size=8, **kw),
        )
        summary_embeddings = embedder.embed_for_clustering(summaries, batch_size=8)

    progress("embed", f"Embedded: {chunk_embeddings.shape[0]} chunks + {summary_embeddings.shape[0]} summaries ({embedder.dim}-dim)")
    stage_end("embed")

    # Persist chunks to DuckDB (batch)
    stage_start("persist_db")
    progress("persist", "Writing chunks to DuckDB...")
    for i, chunk in enumerate(all_chunks):
        store.insert_chunk(
            chunk["chunk_id"], chunk["agreement_id"], chunk["text"],
            chunk["chunk_index"], chunk_embeddings[i], chunk.get("token_count", 0),
        )

    stage_end("persist_db")

    # Persist to LanceDB
    stage_start("persist_lance")
    progress("persist", "Writing vectors to LanceDB...")
    lance_store.upsert_chunks(
        chunk_ids=chunk_ids, vectors=chunk_embeddings,
        texts=chunk_texts, agreement_ids=chunk_agr_ids,
    )
    lance_store.upsert_documents(
        agreement_ids=[a["agreement_id"] for a in agreements],
        vectors=summary_embeddings,
        filenames=[a.get("filename", "") for a in agreements],
        summaries=summaries,
    )
    stage_end("persist_lance")
    store.update_run(run_id, "embed_complete")

    # Save embeddings
    _save_embeddings(run_dir, chunk_embeddings, summary_embeddings)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 4: Macro clustering
    # ══════════════════════════════════════════════════════════════════════
    stage_start("macro")
    progress("macro", f"Macro clustering {len(agreements)} documents...")
    n_docs = len(summaries)
    _calc_probs = cuml_available()  # BERTopic tries cuml import for probabilities

    # Need at least 10 docs for meaningful UMAP + HDBSCAN clustering
    if n_docs < 10:
        progress("macro", f"Small corpus ({n_docs} docs) — assigning all to single domain")
        macro_topics = [0] * n_docs
        macro_model = None
        unique_macro = [0]
        n_domains = 1
        outlier_mask = []
        n_outliers = 0

        domain_id = f"d_{uuid.uuid4().hex[:8]}"
        centroid = summary_embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        domain_map = {0: {
            "domain_id": domain_id, "topic_id": 0,
            "label": "All Documents", "description": "Single domain (too few docs to cluster)",
            "agreement_count": n_docs, "centroid": centroid,
            "agreement_indices": list(range(n_docs)),
        }}
        store.upsert_domain(domain_id, "All Documents", "Single domain", n_docs,
                            float(np.mean([np.dot(summary_embeddings[i], centroid) for i in range(n_docs)])), centroid)
        for i in range(n_docs):
            conf = float(np.dot(summary_embeddings[i], centroid))
            store.update_agreement_domain(agreements[i]["agreement_id"], domain_id, conf)
    else:
        # ── Semi-supervised UMAP with KeyBERT labels ──
        macro_target_weight = 0.0
        if keybert_scorer and KEYBERT_UMAP_TARGET_WEIGHT > 0:
            macro_target_weight = KEYBERT_UMAP_TARGET_WEIGHT
            progress("macro", f"Using semi-supervised UMAP (target_weight={macro_target_weight})")

        macro_umap = make_umap(
            n_neighbors=min(15, n_docs - 1),
            n_components=min(5, max(2, n_docs - 2)),
            min_dist=0.0, metric="cosine",
            target_weight=macro_target_weight,
        )
        macro_hdbscan = make_hdbscan(
            min_cluster_size=max(MACRO_MIN_CLUSTER_SIZE, max(3, n_docs // 50)),
            min_samples=MACRO_MIN_SAMPLES,
            metric="euclidean", cluster_selection_method="eom",
        )
        # Legacy-matching vectorizer: picklable analyzer, (1,4) n-grams, 100k features
        macro_analyzer = StopwordFilteringAnalyzer(
            ngram_range=(1, 4), stopwords=COMBINED_STOPWORDS,
        )
        macro_vectorizer = CountVectorizer(
            analyzer=macro_analyzer,
            max_features=100000,
            min_df=1,
        )
        # BM25-weighted c-TF-IDF (legacy: amplifies domain-specific terms)
        macro_ctfidf = ClassTfidfTransformer(
            bm25_weighting=True, reduce_frequent_words=True,
        )
        macro_model = BERTopic(
            umap_model=macro_umap, hdbscan_model=macro_hdbscan,
            vectorizer_model=macro_vectorizer,
            ctfidf_model=macro_ctfidf,
            nr_topics="auto" if n_docs >= 50 else None,
            calculate_probabilities=_calc_probs, verbose=False,
        )
        # Semi-supervised: pass KeyBERT labels as y if available
        macro_y = None
        if keybert_scorer and macro_target_weight > 0:
            macro_y = keybert_scorer.batch_labels(summaries)
            n_labeled = int((macro_y >= 0).sum())
            progress("macro", f"Semi-supervised: {n_labeled}/{len(summaries)} docs labeled by KeyBERT")

        macro_topics, _ = macro_model.fit_transform(summaries, summary_embeddings, y=macro_y)

        # Save macro model
        _save_model(macro_model, run_dir, "macro_model")

        # Build domain records
        unique_macro = sorted(set(t for t in macro_topics if t != -1))
        domain_map: dict[int, dict] = {}

        for topic_id in unique_macro:
            mask = [i for i, t in enumerate(macro_topics) if t == topic_id]
            centroid = summary_embeddings[mask].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            info = _auto_label_topic(macro_model, topic_id)
            domain_id = f"d_{uuid.uuid4().hex[:8]}"

            domain_map[topic_id] = {
                "domain_id": domain_id, "topic_id": topic_id,
                "label": info["label"], "description": info["description"],
                "agreement_count": len(mask), "centroid": centroid,
                "agreement_indices": mask,
            }

            store.upsert_domain(
                domain_id, info["label"], info["description"],
                len(mask), float(np.mean([np.dot(summary_embeddings[i], centroid) for i in mask])),
                centroid,
            )
            for idx in mask:
                conf = float(np.dot(summary_embeddings[idx], centroid))
                store.update_agreement_domain(agreements[idx]["agreement_id"], domain_id, conf)

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
        n_outliers = len(outlier_mask)
    progress("macro", f"{n_domains} domains, {n_outliers} outlier docs")
    stage_end("macro")
    store.update_run(run_id, "macro_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 5: Micro clustering per domain
    # ══════════════════════════════════════════════════════════════════════
    stage_start("micro")
    progress("micro", "Micro-clustering within each domain...")
    merger = HierarchicalTopicMerger()
    total_clusters = 0
    quality_report: dict = {"domains": {}, "global": {}}  # quality eval accumulator
    micro_models_dir = run_dir / "micro_models"

    for topic_id, dr in domain_map.items():
        if topic_id == -1:
            continue
        domain_id = dr["domain_id"]
        domain_label = dr["label"]

        domain_agr_ids = set(agreements[i]["agreement_id"] for i in dr["agreement_indices"])
        domain_chunk_idx = [
            i for i, c in enumerate(all_chunks) if c["agreement_id"] in domain_agr_ids
        ]

        if len(domain_chunk_idx) < 5:
            progress("micro", f"  {domain_label}: skipped ({len(domain_chunk_idx)} chunks)")
            continue

        d_texts = [all_chunks[i]["text"] for i in domain_chunk_idx]
        d_ids = [all_chunks[i]["chunk_id"] for i in domain_chunk_idx]
        d_embs = chunk_embeddings[domain_chunk_idx]

        n = len(d_texts)

        # ── Bayesian optimization (30 trials, sequential Optuna TPE) for ≥20 chunks ──
        if n >= 20:
            progress("micro", f"  {domain_label}: optimizing ({n} chunks, {OPTIMIZE_TRIALS} trials)...")
            domain_min_clusters = max(3, n // 10)

            from core.ray_optimizer import optimize_sequential
            opt_result = optimize_sequential(
                docs=d_texts,
                embeddings=d_embs,
                n_trials=OPTIMIZE_TRIALS,
                stopwords=list(COMBINED_STOPWORDS),
                min_clusters=domain_min_clusters,
            )

            best_params = opt_result.best_params
            progress("micro", f"  {domain_label}: best_score={opt_result.best_score:.4f}")

            # Rebuild best model with full config for topic extraction
            n_neighbors = min(int(best_params.get("umap_n_neighbors", UMAP_N_NEIGHBORS)), n - 1)
            n_components = max(2, min(int(best_params.get("umap_n_components", UMAP_N_COMPONENTS)), n - 2, n_neighbors - 1))

            micro_target_weight = KEYBERT_UMAP_TARGET_WEIGHT if keybert_scorer else 0.0
            micro_umap = make_umap(
                n_neighbors=n_neighbors, n_components=n_components,
                min_dist=float(best_params.get("umap_min_dist", UMAP_MIN_DIST)),
                metric="cosine",
                target_weight=micro_target_weight,
            )
            micro_hdbscan = make_hdbscan(
                min_cluster_size=max(2, int(best_params.get("hdbscan_min_cluster_size", MICRO_MIN_CLUSTER_SIZE))),
                min_samples=max(1, int(best_params.get("hdbscan_min_samples", MICRO_MIN_SAMPLES))),
                metric="euclidean", cluster_selection_method="eom",
            )
        else:
            # Small domain: direct BERTopic without optimization
            n_neighbors = min(UMAP_N_NEIGHBORS, n - 1)
            n_components = max(2, min(UMAP_N_COMPONENTS, n - 2, n_neighbors - 1))

            micro_target_weight = KEYBERT_UMAP_TARGET_WEIGHT if keybert_scorer else 0.0
            micro_umap = make_umap(
                n_neighbors=n_neighbors, n_components=n_components,
                min_dist=UMAP_MIN_DIST, metric="cosine",
                target_weight=micro_target_weight,
            )
            min_cs = max(MICRO_MIN_CLUSTER_SIZE, max(2, n // 20))
            micro_hdbscan = make_hdbscan(
                min_cluster_size=min_cs,
                min_samples=max(MICRO_MIN_SAMPLES, 1),
                metric="euclidean", cluster_selection_method="eom",
            )

        # Legacy-matching vectorizer + c-TF-IDF (shared by both paths)
        micro_analyzer = StopwordFilteringAnalyzer(
            ngram_range=(1, 4), stopwords=COMBINED_STOPWORDS,
        )
        micro_vectorizer = CountVectorizer(
            analyzer=micro_analyzer,
            max_features=100000,
            min_df=1,  # BERTopic aggregates docs-per-topic; effective doc count is low
        )
        micro_ctfidf = ClassTfidfTransformer(
            bm25_weighting=True, reduce_frequent_words=True,
        )
        micro_model = BERTopic(
            umap_model=micro_umap, hdbscan_model=micro_hdbscan,
            vectorizer_model=micro_vectorizer,
            ctfidf_model=micro_ctfidf,
            calculate_probabilities=_calc_probs, verbose=False,
        )
        # Semi-supervised: pass KeyBERT labels for micro clustering
        micro_y = None
        if keybert_scorer and micro_target_weight > 0:
            micro_y = keybert_scorer.batch_labels(d_texts)

        micro_topics, _ = micro_model.fit_transform(d_texts, d_embs, y=micro_y)

        # Outlier reduction (legacy: c-TF-IDF strategy, threshold=0.3)
        if -1 in micro_topics:
            try:
                new_topics = micro_model.reduce_outliers(d_texts, micro_topics, strategy="c-tf-idf", threshold=0.3)
                # Preserve custom vectorizer on update (legacy pattern)
                micro_model.update_topics(
                    d_texts, topics=new_topics,
                    vectorizer_model=micro_model.vectorizer_model,
                    representation_model=None,
                )
                micro_topics = new_topics
            except Exception:
                pass

        # Save micro model
        _save_model(micro_model, micro_models_dir, domain_id)

        # Quality + records
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
                "cluster_id": cluster_id, "topic_id": tid,
                "domain_id": domain_id, "label": info["label"],
                "description": info["description"], "keywords": keywords,
                "chunk_count": len(mask), "quality_score": quality.get(tid, 0.0),
                "centroid": centroid, "chunk_ids": [d_ids[i] for i in mask],
            })

        if len(cluster_records) > 1:
            cluster_records = merger.merge_similar_topics(cluster_records)

        # Persist
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
                cr.get("keywords", []), cr["chunk_count"], len(c_agr_ids),
                cr.get("quality_score"), cr.get("centroid"),
                cr.get("aliases", []), cr.get("merged_from", []),
            )
            if cr.get("centroid") is not None:
                store.save_centroid(cr["cluster_id"], cr["centroid"], cr["chunk_count"])
            for cid in c_chunk_ids:
                store.insert_cluster_assignment(cid, cr["cluster_id"])

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
        progress("micro", f"  {domain_label}: {n_micro} clusters from {n} chunks")

        # Accumulate quality report per domain
        n_outlier_chunks = sum(1 for t in micro_topics if t == -1)
        domain_quality_scores = [q for q in quality.values() if q > 0]
        quality_report["domains"][domain_id] = {
            "label": domain_label,
            "n_chunks": n,
            "n_clusters": n_micro,
            "n_outlier_chunks": n_outlier_chunks,
            "outlier_ratio": round(n_outlier_chunks / n, 4) if n > 0 else 0,
            "avg_quality": round(float(np.mean(domain_quality_scores)), 4) if domain_quality_scores else 0,
            "min_quality": round(float(min(domain_quality_scores)), 4) if domain_quality_scores else 0,
            "max_quality": round(float(max(domain_quality_scores)), 4) if domain_quality_scores else 0,
            "clusters": [
                {
                    "cluster_id": cr["cluster_id"],
                    "label": cr["label"],
                    "chunk_count": cr["chunk_count"],
                    "quality_score": round(cr.get("quality_score", 0), 4),
                    "keywords": cr.get("keywords", [])[:10],
                }
                for cr in cluster_records
            ],
        }

    stage_end("micro")

    # Compute global quality summary
    all_domain_avgs = [d["avg_quality"] for d in quality_report["domains"].values() if d["avg_quality"] > 0]
    total_outlier_chunks = sum(d["n_outlier_chunks"] for d in quality_report["domains"].values())
    total_domain_chunks = sum(d["n_chunks"] for d in quality_report["domains"].values())
    quality_report["global"] = {
        "n_documents": len(agreements),
        "n_chunks": len(all_chunks),
        "n_domains": n_domains,
        "n_clusters": total_clusters,
        "n_outlier_docs": n_outliers,
        "n_outlier_chunks": total_outlier_chunks,
        "chunk_outlier_ratio": round(total_outlier_chunks / total_domain_chunks, 4) if total_domain_chunks > 0 else 0,
        "avg_quality_across_domains": round(float(np.mean(all_domain_avgs)), 4) if all_domain_avgs else 0,
        "use_semantic_chunking": use_semantic_chunking,
        "device": device,
        "cuml_available": cuml_available(),
    }

    # Save quality report
    _save_quality_report(run_dir, quality_report)
    progress("quality", f"Quality report saved ({total_clusters} clusters, avg={quality_report['global']['avg_quality_across_domains']:.4f})")

    store.update_run(run_id, "micro_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 5.5: ARM Enrichment (Pass 2)
    # ══════════════════════════════════════════════════════════════════════
    arm_result = {"n_rules": 0, "n_packages": 0}
    if ENABLE_ARM_ENRICHMENT:
        stage_start("arm")
        progress("arm", "Mining clause association rules (FP-Growth)...")
        try:
            from core.relationship_layer import RelationshipLayer
            from core.arm.arm_miner import mlxtend_available

            if mlxtend_available():
                relationship_layer = RelationshipLayer(store)

                # Build clause assignments: {doc_id: [clause_types]}
                # Each document's clusters represent its clause types
                doc_clause_types: dict[str, list[str]] = {}
                for topic_id, dr in domain_map.items():
                    if topic_id == -1:
                        continue
                    domain_agr_ids = set(agreements[i]["agreement_id"] for i in dr["agreement_indices"])
                    domain_chunk_idx = [
                        i for i, c in enumerate(all_chunks) if c["agreement_id"] in domain_agr_ids
                    ]
                    # Get cluster labels for chunks in this domain
                    for cr in quality_report["domains"].get(dr["domain_id"], {}).get("clusters", []):
                        clause_label = cr.get("label", "Unknown")
                        # Find which agreements have chunks in this cluster
                        for c in all_chunks:
                            if c["agreement_id"] in domain_agr_ids:
                                doc_id = c["agreement_id"]
                                if doc_id not in doc_clause_types:
                                    doc_clause_types[doc_id] = []
                                if clause_label not in doc_clause_types[doc_id]:
                                    doc_clause_types[doc_id].append(clause_label)

                if doc_clause_types:
                    relationship_layer.mine_from_assignments(
                        doc_clause_types, pipeline_run_id=run_id
                    )

                    # Count results
                    rules = relationship_layer._load_rules()
                    packages = relationship_layer.get_term_packages()
                    arm_result = {"n_rules": len(rules), "n_packages": len(packages)}
                    progress("arm", f"ARM: {len(rules)} rules, {len(packages)} term packages")

                    # Save ARM artifacts
                    import json as _json
                    arm_rules_path = run_dir / "arm_rules.json"
                    arm_packages_path = run_dir / "term_packages.json"
                    with open(arm_rules_path, "w") as f:
                        _json.dump(rules, f, indent=2, default=str)
                    with open(arm_packages_path, "w") as f:
                        _json.dump(packages, f, indent=2, default=str)
                else:
                    progress("arm", "No clause assignments to mine")
            else:
                progress("arm", "Skipped (mlxtend not installed)")
        except Exception as e:
            logger.error(f"ARM enrichment failed: {e}")
            progress("arm", f"Failed: {e}")
        stage_end("arm")
    else:
        progress("arm", "Skipped (ENABLE_ARM_ENRICHMENT=false)")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 6: Field Discovery (Azure OpenAI)
    # ══════════════════════════════════════════════════════════════════════
    field_discovery_result = {"total_fields": 0, "total_extractions": 0}
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        stage_start("field_discovery")
        progress("field_discovery", "Discovering fields via Azure OpenAI...")
        try:
            from core.field_discovery import run_field_discovery, FieldDiscoveryConfig
            fd_config = FieldDiscoveryConfig(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4"),
            )
            field_discovery_result = run_field_discovery(
                store=store, config=fd_config,
                progress_fn=progress, extract_values=True,
                min_cluster_chunks=5,
                use_rlm=use_rlm,
            )
        except Exception as e:
            logger.error(f"Field discovery failed: {e}")
            progress("field_discovery", f"Failed: {e}")
        stage_end("field_discovery")
    else:
        progress("field_discovery", "Skipped (AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY not set)")

    store.update_run(run_id, "completed", "completed")

    # ══════════════════════════════════════════════════════════════════════
    # Save manifest
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    stats = store.get_stats()
    manifest = {
        "run_id": run_id,
        "account_id": account_id,
        "pdf_dir": str(pdf_dir),
        "n_documents": len(agreements),
        "n_chunks": len(all_chunks),
        "n_domains": n_domains,
        "n_clusters": total_clusters,
        "n_outlier_docs": n_outliers,
        "embedding_dim": int(embedder.dim),
        "device": device,
        "cuml_available": cuml_available(),
        "use_semantic_chunking": use_semantic_chunking,
        "elapsed_seconds": round(elapsed, 1),
        "stage_timings": stage_timings,
        "db_path": str(db),
        "lance_path": str(lance),
        "artifacts_dir": str(run_dir),
        "macro_model_path": str(run_dir / "macro_model"),
        "micro_models_dir": str(micro_models_dir),
        "embeddings_path": str(run_dir / "embeddings.npz"),
        "quality_report_path": str(run_dir / "cluster_quality.json"),
        "field_discovery": field_discovery_result,
        "arm_enrichment": arm_result,
        "stats": stats,
    }
    _save_manifest(run_dir, manifest)

    progress("done", f"Complete in {elapsed:.0f}s — {len(agreements)} docs, "
             f"{len(all_chunks)} chunks, {n_domains} domains, {total_clusters} clusters")
    progress("done", f"Artifacts: {run_dir}")

    return {
        "store": store,
        "lance_store": lance_store,
        "run_id": run_id,
        "account_id": account_id,
        "stats": stats,
        "n_documents": len(agreements),
        "n_chunks": len(all_chunks),
        "n_domains": n_domains,
        "n_clusters": total_clusters,
        "elapsed_seconds": round(elapsed, 1),
        "artifacts_dir": str(run_dir),
        "manifest": manifest,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Run featurizer + clustering pipeline on PDFs")
    parser.add_argument("pdf_dir", help="Directory containing PDF files")
    parser.add_argument("--account-id", default="default", help="Account/tenant ID")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--fast-chunk", action="store_true", help="Use sliding window instead of semantic chunking")
    args = parser.parse_args()

    result = run_cluster_pipeline(
        pdf_dir=args.pdf_dir,
        account_id=args.account_id,
        max_docs=args.max_docs,
        device=args.device,
        use_semantic_chunking=not args.fast_chunk,
    )

    print(f"\n{'='*60}")
    print(f"  Documents:  {result['n_documents']}")
    print(f"  Chunks:     {result['n_chunks']}")
    print(f"  Domains:    {result['n_domains']}")
    print(f"  Clusters:   {result['n_clusters']}")
    print(f"  Time:       {result['elapsed_seconds']}s")
    print(f"  Artifacts:  {result['artifacts_dir']}")
    print(f"{'='*60}")
