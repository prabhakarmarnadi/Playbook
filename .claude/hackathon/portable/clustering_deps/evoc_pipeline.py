"""
EVoC (Embedding Vector Oriented Clustering) pipeline.

Replaces UMAP + HDBSCAN with EVoC, which is purpose-built for high-dimensional
embedding vectors.  Supports three modes:

  --mode macro-micro   Document-level macro → per-domain chunk-level micro
  --mode clause        Flat clause-type clustering (like clause_pipeline.py)
  --mode hybrid        Clause-type clustering + document-domain macro clustering

EVoC advantages over HDBSCAN:
  - No UMAP reduction needed (works directly on embeddings)
  - Multi-granularity cluster layers built in
  - Fewer hyperparameters to tune
  - Designed for cosine similarity on embeddings

Optuna integration:
  - Tunes base_min_cluster_size, noise_level, n_neighbors, min_samples
  - Scores each layer independently, picks best layer per trial
  - Uses same scoring engine as HDBSCAN pipeline (cosine + silhouette + DBCV)

Usage:
    python -m core.evoc_pipeline /path/to/pdfs --account-id acct_001 --mode clause
    python -m core.evoc_pipeline /path/to/pdfs --account-id acct_001 --mode macro-micro
    python -m core.evoc_pipeline /path/to/pdfs --account-id acct_001 --mode hybrid
"""
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import evoc

from config import (
    DB_PATH, DATA_DIR, LEGAL_STOPWORDS,
    MACRO_MIN_CLUSTER_SIZE, MACRO_MIN_SAMPLES,
    MICRO_MIN_CLUSTER_SIZE, MICRO_MIN_SAMPLES,
    UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS, UMAP_MIN_DIST,
    OPTIMIZE_TRIALS,
)
from core.pdf_parser import parse_pdf_directory, load_parsed_agreements, PARSED_TEXT_DIR
from core.chunker import semantic_chunk, sliding_window_chunk, get_document_summary, SemanticChunkerConfig, extract_clause_header
from core.nomic_embedder import NomicEmbedder
from core.cuml_wrappers import cuml_available
from core.lancedb_store import LanceVectorStore
from core.store import ClusteringStore
from core.topic_merger import HierarchicalTopicMerger
from core.micro_clusterer import compute_cluster_quality

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = DATA_DIR / "artifacts"


# ── EVoC Wrappers ──────────────────────────────────────────────────────────────

# EVoC parameter ranges for Optuna tuning
EVOC_PARAM_RANGES = {
    "base_min_cluster_size": (3, 25),
    "noise_level": (0.1, 0.6),
    "n_neighbors": (5, 30),
    "min_samples": (1, 8),
}


def evoc_cluster(
    embeddings: np.ndarray,
    base_min_cluster_size: int = 10,
    noise_level: float = 0.3,
    n_neighbors: int = 15,
    min_samples: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Run EVoC clustering on embedding vectors.

    Returns:
        (labels, cluster_layers, persistence_scores)
    """
    n = embeddings.shape[0]
    if n < 5:
        return np.zeros(n, dtype=int), [np.zeros(n, dtype=int)], [1.0]

    model = evoc.EVoC(
        base_min_cluster_size=min(base_min_cluster_size, max(2, n // 10)),
        noise_level=noise_level,
        n_neighbors=min(n_neighbors, n - 1),
        min_samples=max(1, min(min_samples, n // 20)),
        n_epochs=50,
        random_state=random_state,
    )
    labels = model.fit_predict(embeddings.astype(np.float32))
    layers = model.cluster_layers_ if hasattr(model, 'cluster_layers_') else [labels]
    scores = model.persistence_scores_ if hasattr(model, 'persistence_scores_') else [1.0]
    return labels, layers, list(scores)


def _select_best_layer(
    layers: list[np.ndarray],
    embeddings: np.ndarray,
    persistence_scores: list[float],
    prefer_granularity: bool = False,
) -> tuple[np.ndarray, int, list[dict]]:
    """Select the best EVoC layer using scoring metrics instead of just persistence.

    Evaluates each layer via silhouette + cosine coherence + outlier ratio,
    rather than relying solely on EVoC's persistence_scores_ (which tends to
    pick overly coarse layers for tightly-packed legal embeddings).

    Args:
        prefer_granularity: If True, add a bonus for layers with more clusters
            (useful for macro domain clustering where we want 10-15 domains,
            not 3-5).

    Returns: (best_labels, best_layer_index, layer_scores)
        layer_scores is a list of dicts with silhouette/cosine/composite per layer.
    """
    layer_scores = [{} for _ in layers]

    if len(layers) == 1:
        return layers[0], 0, layer_scores

    from core.scoring import calculate_scores
    best_score = -999.0
    best_idx = 0

    # Find max cluster count across valid layers for normalization
    max_types = max(
        (len(set(layer) - {-1}) for layer in layers
         if len(set(layer) - {-1}) >= 2),
        default=2,
    )

    for li, layer in enumerate(layers):
        n_types = len(set(layer) - {-1})
        n_out = sum(1 for t in layer if t == -1)
        outlier_ratio = n_out / len(layer) if len(layer) > 0 else 1.0

        # Skip layers with no clusters or extreme outlier rates
        if n_types < 2 or outlier_ratio > 0.6:
            layer_scores[li] = {"silhouette": None, "cosine": None, "composite": None}
            continue

        topics = np.array(layer)
        scores = calculate_scores(topics, embeddings)

        if prefer_granularity:
            # For macro: reward more domains, reduce silhouette dominance
            granularity_bonus = (n_types / max_types) if max_types > 0 else 0
            composite = (
                scores.silhouette_score * 0.25
                + scores.average_cosine_similarity * 0.25
                + (1.0 - outlier_ratio) * 0.25
                + granularity_bonus * 0.25
            )
        else:
            composite = (
                scores.silhouette_score * 0.4
                + scores.average_cosine_similarity * 0.3
                + (1.0 - outlier_ratio) * 0.3
            )

        layer_scores[li] = {
            "silhouette": round(scores.silhouette_score, 4),
            "cosine": round(scores.average_cosine_similarity, 4),
            "composite": round(composite, 4),
        }

        logger.info(f"  Layer {li}: {n_types} types, {outlier_ratio:.1%} outliers, "
                     f"sil={scores.silhouette_score:.4f}, cos={scores.average_cosine_similarity:.4f}, "
                     f"composite={composite:.4f}")

        if composite > best_score:
            best_score = composite
            best_idx = li

    return layers[best_idx], best_idx, layer_scores


def _evoc_build_and_score(
    params: dict,
    embeddings: np.ndarray,
    min_clusters: int = 3,
) -> tuple[float, np.ndarray, int]:
    """Build EVoC model with given params, score all layers, return best.

    Returns: (score, best_labels, best_layer_idx)
    """
    from core.scoring import calculate_scores, apply_penalties

    n = embeddings.shape[0]
    mcs = max(2, min(int(params["base_min_cluster_size"]), n // 10))
    nn = min(int(params["n_neighbors"]), n - 1)
    ms = max(1, min(int(params["min_samples"]), n // 20))

    model = evoc.EVoC(
        base_min_cluster_size=mcs,
        noise_level=float(params["noise_level"]),
        n_neighbors=nn,
        min_samples=ms,
        n_epochs=50,
        random_state=42,
    )
    labels = model.fit_predict(embeddings.astype(np.float32))
    layers = model.cluster_layers_ if hasattr(model, 'cluster_layers_') else [labels]
    persistence = model.persistence_scores_ if hasattr(model, 'persistence_scores_') else [1.0]

    # Score each layer, return best
    best_score = -999.0
    best_labels = labels
    best_layer_idx = 0

    for li, layer in enumerate(layers):
        topics = np.array(layer)
        n_types = len(set(topics) - {-1})
        n_out = int(np.sum(topics == -1))
        outlier_ratio = n_out / len(topics) if len(topics) > 0 else 1.0

        if n_types < 2 or outlier_ratio > 0.6:
            continue

        scores = calculate_scores(topics, embeddings)
        scores = apply_penalties(
            scores,
            min_clusters=min_clusters,
            max_dominance=0.15,
            max_outlier_ratio=0.25,
        )

        if scores.adjusted_score > best_score:
            best_score = scores.adjusted_score
            best_labels = layer
            best_layer_idx = li

    return best_score, best_labels, best_layer_idx


def evoc_optimize(
    embeddings: np.ndarray,
    n_trials: int = 30,
    param_ranges: dict | None = None,
    min_clusters: int = 3,
    progress_fn=None,
) -> tuple[dict, np.ndarray, float]:
    """Optuna-driven EVoC hyperparameter optimization.

    Tunes base_min_cluster_size, noise_level, n_neighbors, min_samples.
    Scores each layer independently and picks the best.

    Returns: (best_params, best_labels, best_score)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ranges = param_ranges or EVOC_PARAM_RANGES
    progress = progress_fn or (lambda s, m: None)

    best_labels_overall = None
    best_score_overall = -999.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_labels_overall, best_score_overall
        params = {}
        for key, (lo, hi) in ranges.items():
            if isinstance(lo, int) and isinstance(hi, int):
                params[key] = trial.suggest_int(key, lo, hi)
            else:
                params[key] = trial.suggest_float(key, lo, hi)

        try:
            score, labels, layer_idx = _evoc_build_and_score(
                params, embeddings, min_clusters=min_clusters,
            )
            if score > best_score_overall:
                best_score_overall = score
                best_labels_overall = labels.copy()

            return score
        except Exception as e:
            logger.warning(f"EVoC trial failed: {e}")
            return -1.0

    sampler = optuna.samplers.TPESampler(n_startup_trials=min(8, n_trials // 3), seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    progress("optuna", f"EVoC optimization: {n_trials} trials, "
             f"best_score={study.best_value:.4f}, params={best_params}")

    # Final run with best params to get labels
    if best_labels_overall is None:
        _, best_labels_overall, _ = _evoc_build_and_score(
            best_params, embeddings, min_clusters=min_clusters,
        )

    return best_params, best_labels_overall, study.best_value


def evoc_cluster_with_target(
    embeddings: np.ndarray,
    target_clusters: int | None = None,
    noise_level: float = 0.3,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run EVoC with an approximate target number of clusters.

    Returns:
        (labels, cluster_layers)
    """
    n = embeddings.shape[0]
    if n < 5:
        return np.zeros(n, dtype=int), [np.zeros(n, dtype=int)]

    kwargs = dict(
        noise_level=noise_level,
        n_neighbors=min(n_neighbors, n - 1),
        min_samples=max(1, min(3, n // 20)),
        n_epochs=50,
        random_state=random_state,
    )
    if target_clusters:
        kwargs["approx_n_clusters"] = target_clusters
    else:
        kwargs["base_min_cluster_size"] = max(2, n // 100)

    model = evoc.EVoC(**kwargs)
    labels = model.fit_predict(embeddings.astype(np.float32))
    layers = model.cluster_layers_ if hasattr(model, 'cluster_layers_') else [labels]
    return labels, layers


# ── LLM Cluster Labeling (shared with clause_pipeline) ────────────────────────

def _get_labeling_client():
    """Create Azure OpenAI client for labeling."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if not endpoint or not api_key:
        return None
    try:
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    except Exception:
        return None


def _llm_label_call(client, deployment: str, prompt: str) -> dict | None:
    """Single Azure OpenAI call for labeling. Returns parsed JSON or None."""
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=500,
                response_format={"type": "json_object"},
                timeout=30,
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            logger.warning(f"LLM label attempt {attempt+1} failed: {e}")
            if attempt < 1:
                time.sleep(1)
    return None


def _llm_label_clusters(
    cluster_labels: np.ndarray,
    texts: list[str],
    context_label: str = "legal agreements",
) -> dict[int, dict]:
    """Label clusters via LLM using representative texts.

    Works without BERTopic — just uses the cluster assignments and raw texts.
    """
    client = _get_labeling_client()
    if not client:
        return {}

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    labels = {}
    unique_topics = sorted(set(t for t in cluster_labels if t != -1))

    for tid in unique_topics:
        mask = [i for i, t in enumerate(cluster_labels) if t == tid]
        # Sample up to 5 representative texts
        sample_indices = mask[:5]
        sample_texts = [texts[i][:400] for i in sample_indices]

        prompt = f"""You are labeling clusters of text from {context_label}.

This cluster has {len(mask)} items. Here are representative excerpts:
{chr(10).join(f'- {t}' for t in sample_texts)}

What type of content is in this cluster?

Respond in JSON:
{{"label": "A clear 2-5 word descriptive name", "description": "One sentence describing what this cluster covers"}}"""

        result = _llm_label_call(client, deployment, prompt)
        if result and result.get("label"):
            labels[tid] = result
            logger.info(f"  Cluster {tid}: '{result['label']}'")
        else:
            labels[tid] = {"label": f"Cluster_{tid}", "description": ""}

    return labels


# ── Utility functions ──────────────────────────────────────────────────────────

def _save_embeddings(path: Path, chunk_embeddings: np.ndarray, summary_embeddings: np.ndarray):
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "embeddings.npz",
        chunk_embeddings=chunk_embeddings,
        summary_embeddings=summary_embeddings,
    )


def _load_cached_embeddings(account_dir: Path, expected_chunks: int, expected_summaries: int):
    if not account_dir.exists():
        return None, None
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
            logger.info(f"Embedding cache HIT: {cache_path}")
            return chunk_emb, summary_emb
    except Exception:
        pass
    return None, None


def _save_manifest(path: Path, manifest: dict):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def _save_quality_report(path: Path, report: dict):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "cluster_quality.json", "w") as f:
        json.dump(report, f, indent=2, default=str)


def _dedup_universal_fields(store, progress):
    from collections import Counter
    all_fields = store.get_fields()
    field_key_counts = Counter()
    field_key_ids = {}
    for f in all_fields:
        key = (f["name"], f.get("field_type", "text"))
        field_key_counts[key] += 1
        field_key_ids.setdefault(key, []).append(f["field_id"])

    universal_fields = [(name, ftype, ids) for (name, ftype), ids
                        in field_key_ids.items() if field_key_counts[(name, ftype)] >= 2]
    if universal_fields:
        for name, ftype, ids in universal_fields:
            for fid in ids:
                try:
                    store.conn.execute(
                        "UPDATE field_definitions SET source = 'universal' WHERE field_id = ?",
                        [fid]
                    )
                except Exception:
                    pass
        progress("field_dedup", f"Tagged {len(universal_fields)} universal fields")


def _export_fields_json(store, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    domains = {d["domain_id"]: d for d in store.get_domains()}
    clusters = {c["cluster_id"]: c for c in store.get_clusters()}
    fields = []
    for fd in store.get_fields():
        cluster = clusters.get(fd.get("cluster_id", ""), {})
        domain = domains.get(cluster.get("domain_id", ""), {})
        examples = fd.get("examples", [])
        if isinstance(examples, str):
            try:
                examples = json.loads(examples)
            except Exception:
                examples = []
        fields.append({
            "field_id": fd.get("field_id"),
            "name": fd.get("name"),
            "field_type": fd.get("field_type"),
            "description": fd.get("description"),
            "source": fd.get("source"),
            "examples": examples,
            "cluster_id": fd.get("cluster_id"),
            "cluster_label": cluster.get("label", ""),
            "domain_id": cluster.get("domain_id", ""),
            "domain_label": domain.get("label", ""),
        })
    out = path / "fields.json"
    with open(out, "w") as f:
        json.dump({"total_fields": len(fields), "fields": fields}, f, indent=2, default=str)


def _export_chunks_json(store, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    domains = {d["domain_id"]: d for d in store.get_domains()}
    clusters = {c["cluster_id"]: c for c in store.get_clusters()}
    agreements = {a["agreement_id"]: a for a in store.get_agreements()}
    rows = store.conn.execute("""
        SELECT ch.chunk_id, ch.agreement_id, ch.chunk_text, ch.chunk_index, ch.token_count,
               ca.cluster_id
        FROM chunks ch
        LEFT JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
        ORDER BY ch.agreement_id, ch.chunk_index
    """).fetchall()
    chunks = []
    for chunk_id, agr_id, text, cidx, tcnt, cid in rows:
        cluster = clusters.get(cid or "", {})
        domain = domains.get(cluster.get("domain_id", ""), {})
        agr = agreements.get(agr_id or "", {})
        chunks.append({
            "chunk_id": chunk_id, "agreement_id": agr_id,
            "filename": agr.get("filename", ""),
            "chunk_index": cidx, "token_count": tcnt, "text": text,
            "cluster_id": cid, "cluster_label": cluster.get("label", ""),
            "domain_id": cluster.get("domain_id", ""),
            "domain_label": domain.get("label", ""),
        })
    out = path / "chunks.json"
    with open(out, "w") as f:
        json.dump({"total_chunks": len(chunks), "chunks": chunks}, f, indent=2, default=str)


def _export_clause_types_json(store, path: Path, progress_fn=None):
    path.mkdir(parents=True, exist_ok=True)
    progress = progress_fn or (lambda s, m: None)

    clusters = store.get_clusters()
    agreements_map = {a["agreement_id"]: a for a in store.get_agreements()}

    all_clauses_rows = store.conn.execute("""
        SELECT clause_id, agreement_id, clause_number, clause_title,
               heading_level, clause_index, clause_type_id, token_count
        FROM clauses ORDER BY agreement_id, clause_index
    """).fetchdf().to_dict("records")

    clauses_by_type = {}
    for cl in all_clauses_rows:
        ctid = cl.get("clause_type_id") or ""
        clauses_by_type.setdefault(ctid, []).append(cl)

    all_fields = store.get_fields()
    fields_by_cluster = {}
    for f in all_fields:
        cid = f.get("cluster_id", "")
        fields_by_cluster.setdefault(cid, []).append(f)

    clause_types = []
    total_clauses = 0
    total_fields = 0

    for cluster in clusters:
        cid = cluster["cluster_id"]
        type_clauses = clauses_by_type.get(cid, [])
        type_fields = fields_by_cluster.get(cid, [])

        clause_entries = []
        for cl in type_clauses:
            agr = agreements_map.get(cl.get("agreement_id", ""), {})
            clause_entries.append({
                "clause_id": cl["clause_id"],
                "agreement_id": cl["agreement_id"],
                "filename": agr.get("filename", ""),
                "clause_number": cl.get("clause_number", ""),
                "clause_title": cl.get("clause_title", ""),
                "heading_level": cl.get("heading_level", 0),
                "clause_index": cl.get("clause_index", 0),
            })

        field_entries = []
        for f in type_fields:
            examples = f.get("examples", [])
            if isinstance(examples, str):
                try:
                    examples = json.loads(examples)
                except Exception:
                    examples = []
            field_entries.append({
                "field_id": f.get("field_id"), "name": f.get("name"),
                "type": f.get("field_type"), "description": f.get("description"),
                "source": f.get("source"), "examples": examples,
            })

        clause_types.append({
            "clause_type_id": cid,
            "label": cluster.get("label", ""),
            "description": cluster.get("description", ""),
            "keywords": cluster.get("keywords", [])[:10],
            "clause_count": len(clause_entries),
            "chunk_count": cluster.get("chunk_count", 0),
            "agreement_count": cluster.get("agreement_count", 0),
            "quality_score": round(cluster.get("quality_score", 0) or 0, 4),
            "fields": field_entries,
            "clauses": clause_entries,
        })
        total_clauses += len(clause_entries)
        total_fields += len(field_entries)

    clause_types.sort(key=lambda x: x["clause_count"], reverse=True)

    out = path / "clause_types.json"
    with open(out, "w") as f:
        json.dump({
            "total_clause_types": len(clause_types),
            "total_clauses": total_clauses,
            "total_fields": total_fields,
            "clause_types": clause_types,
        }, f, indent=2, default=str)
    progress("export", f"Saved clause_types.json: {out} "
             f"({len(clause_types)} types, {total_clauses} clauses, {total_fields} fields)")


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_evoc_pipeline(
    pdf_dir: str | Path,
    account_id: str = "default",
    mode: str = "hybrid",
    db_path: str | Path | None = None,
    lance_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    device: str = "cuda",
    max_docs: int | None = None,
    use_semantic_chunking: bool = True,
    use_llm_summaries: bool = False,
    evoc_noise_level: float = 0.3,
    use_optuna: bool = True,
    optuna_trials: int = 30,
    use_rlm: bool = False,
    callback=None,
) -> dict:
    """
    Parse PDFs → chunk → embed → EVoC cluster → field discovery → export.

    Args:
        mode: 'clause' for flat clause-type clustering,
              'macro-micro' for document-level → per-domain chunk-level,
              'hybrid' for clause-type + document-domain (recommended).
        evoc_noise_level: EVoC noise_level param (0.0=cluster more, 1.0=purer).
        use_optuna: Run Optuna optimization to find best EVoC params.
        optuna_trials: Number of Optuna trials (default 30).
    """
    t0 = time.time()
    stage_timings: dict[str, float] = {}
    _stage_t0: dict[str, float] = {}

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

    # ── Paths ──
    artifacts_root = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    account_dir = artifacts_root / account_id
    db = Path(db_path) if db_path else DATA_DIR / f"{account_id}.duckdb"
    lance = Path(lance_path) if lance_path else DATA_DIR / f"{account_id}.lance"

    store = ClusteringStore(db)
    lance_store = LanceVectorStore(lance)
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    run_dir = account_dir / run_id

    store.start_run(run_id, "init")
    progress("init", f"EVoC pipeline mode={mode} account={account_id} device={device}")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1: Parse PDFs (with caching)
    # ══════════════════════════════════════════════════════════════════════
    stage_start("parse")
    cached = load_parsed_agreements()
    pdf_dir_resolved = str(Path(pdf_dir).resolve())
    manifest_path = PARSED_TEXT_DIR / "manifest.json"
    cache_hit = False
    if cached and manifest_path.exists():
        with open(manifest_path) as _f:
            _manifest = json.load(_f)
        if str(Path(_manifest.get("source_dir", "")).resolve()) == pdf_dir_resolved:
            cache_hit = True

    if cache_hit:
        progress("parse", f"Using {len(cached)} cached parsed PDFs")
        parsed_docs = cached
    else:
        progress("parse", f"Parsing PDFs from {pdf_dir}...")
        parsed_docs = parse_pdf_directory(pdf_dir)

    if max_docs and len(parsed_docs) > max_docs:
        parsed_docs = parsed_docs[:max_docs]

    if not parsed_docs:
        progress("error", "No PDFs parsed")
        return {"store": store, "lance_store": lance_store, "run_id": run_id,
                "stats": {}, "n_documents": 0}

    agreements = [
        {"agreement_id": d["agreement_id"], "filename": d["filename"], "raw_text": d["raw_text"]}
        for d in parsed_docs
        if len(d.get("raw_text", "")) > 100
    ]
    progress("parse", f"{len(agreements)} valid PDFs")
    stage_end("parse")
    store.update_run(run_id, "parse_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2: Chunk (clause-aware)
    # ══════════════════════════════════════════════════════════════════════
    stage_start("chunk")
    embedder = NomicEmbedder.get_instance(device=device)
    embedder.warmup()

    if use_semantic_chunking:
        progress("chunk", "Semantic chunking (clause-aware)...")
        chunker_config = SemanticChunkerConfig(
            buffer_size=1, breakpoint_percentile=90.0,
            min_chunk_chars=100, max_chunk_chars=2000,
            clause_split=True,
        )
        MAX_CHUNKS_PER_DOC = 80

        from core.chunker import _structural_clause_split, _split_sentences, _combine_sentence_buffers, _compute_breakpoints, _assemble_chunks, _approx_token_count, cohesion_subsplit_clauses
        import uuid as _uuid

        doc_clauses = []
        oversized_segments = []
        os_sentence_data = []
        all_buffers = []
        buffer_offsets = []

        _embed_for_cohesion = lambda texts: embedder.embed_for_clustering(texts, batch_size=8)

        for agr in agreements:
            raw_text = agr["raw_text"]
            sections = _structural_clause_split(raw_text, min_clause_chars=chunker_config.min_chunk_chars)
            if not sections:
                sections = [raw_text.strip()] if raw_text.strip() else []
            # Cohesion sub-split: break omnibus sections that contain multiple topics
            pre_split_count = len(sections)
            sections = cohesion_subsplit_clauses(
                sections, embed_fn=_embed_for_cohesion,
                min_clause_chars=chunker_config.min_chunk_chars,
                cohesion_threshold=0.75,
            )
            if len(sections) > pre_split_count:
                progress("chunk", f"  {agr.get('filename', '?')}: cohesion sub-split "
                         f"{pre_split_count} → {len(sections)} clauses")
            doc_clauses.append((agr, sections))

            for ci, section in enumerate(sections):
                if len(section) > chunker_config.max_chunk_chars:
                    sentences = _split_sentences(section)
                    if sentences and len(sentences) > 2:
                        buffers = _combine_sentence_buffers(sentences, chunker_config.buffer_size)
                        oversized_segments.append((len(doc_clauses) - 1, ci, section))
                        os_sentence_data.append((sentences, buffers))
                        buf_start = len(all_buffers)
                        all_buffers.extend(buffers)
                        buffer_offsets.append((buf_start, len(all_buffers)))
                    else:
                        oversized_segments.append((len(doc_clauses) - 1, ci, section))
                        os_sentence_data.append((sentences or [], []))
                        buffer_offsets.append((len(all_buffers), len(all_buffers)))

        progress("chunk", f"Structural split: {sum(len(s) for _, s in doc_clauses)} sections "
                 f"({len(oversized_segments)} need sub-splitting)")

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

        os_sub_chunks = {}
        for os_i, (doc_idx, clause_idx, section_text) in enumerate(oversized_segments):
            sentences, buffers = os_sentence_data[os_i]
            buf_start, buf_end = buffer_offsets[os_i]

            if buf_start < buf_end and sentences and len(sentences) > 2:
                doc_embeddings = all_buffer_embeddings[buf_start:buf_end]
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
                from core.chunker import _force_split
                sub_texts = _force_split(sentences, chunker_config.max_chunk_chars) if sentences else [section_text]

            os_sub_chunks[(doc_idx, clause_idx)] = sub_texts

        all_chunks = []
        all_clauses = []
        for doc_idx, (agr, sections) in enumerate(doc_clauses):
            chunk_idx = 0
            for ci, section in enumerate(sections):
                sub_texts = os_sub_chunks.get((doc_idx, ci), [section])

                header = extract_clause_header(section)
                clause_id = f"cls_{uuid.uuid4().hex[:10]}"
                clause_rec = {
                    "clause_id": clause_id,
                    "agreement_id": agr["agreement_id"],
                    "clause_number": header["clause_number"] if header else "",
                    "clause_title": header["clause_title"] if header else "",
                    "heading_level": header["heading_level"] if header else 0,
                    "clause_index": ci,
                    "full_text": section,
                    "token_count": _approx_token_count(section),
                    "chunk_ids": [],
                }

                for ct in sub_texts:
                    if chunk_idx >= MAX_CHUNKS_PER_DOC:
                        break
                    chk_id = f"chk_{_uuid.uuid4().hex[:10]}"
                    all_chunks.append({
                        "chunk_id": chk_id, "text": ct,
                        "char_start": 0, "char_end": len(ct),
                        "token_count": _approx_token_count(ct),
                        "chunk_index": chunk_idx,
                        "agreement_id": agr["agreement_id"],
                        "clause_id": clause_id,
                    })
                    clause_rec["chunk_ids"].append(chk_id)
                    chunk_idx += 1

                all_clauses.append(clause_rec)
    else:
        progress("chunk", "Sliding window chunking (fast mode)...")
        all_chunks = []
        all_clauses = []
        for agr in agreements:
            chunks = sliding_window_chunk(agr["raw_text"])
            for c in chunks:
                c["agreement_id"] = agr["agreement_id"]
            all_chunks.extend(chunks)

    progress("chunk", f"{len(all_chunks)} chunks from {len(agreements)} docs")
    stage_end("chunk")

    # Store agreements
    for agr in agreements:
        summary = get_document_summary(agr["raw_text"], agr.get("filename", ""))
        store.insert_agreement(agr["agreement_id"], agr.get("filename", ""), agr["raw_text"], summary)
    store.update_run(run_id, "chunk_complete")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 3: Embed
    # ══════════════════════════════════════════════════════════════════════
    stage_start("embed")
    chunk_texts = [c["text"] for c in all_chunks]
    chunk_ids = [c["chunk_id"] for c in all_chunks]
    chunk_agr_ids = [c["agreement_id"] for c in all_chunks]

    from core.macro_clusterer import get_field_summaries
    _embed_fn = lambda texts, **kw: embedder.embed_for_clustering(texts, batch_size=8, **kw)

    cached_chunk_emb, cached_summary_emb = _load_cached_embeddings(
        account_dir, expected_chunks=len(chunk_texts), expected_summaries=len(agreements),
    )

    if cached_chunk_emb is not None and cached_summary_emb is not None and not use_llm_summaries:
        progress("embed", f"Using cached embeddings")
        chunk_embeddings = cached_chunk_emb
        summary_embeddings = cached_summary_emb
        summaries = get_field_summaries(agreements, all_chunks, chunk_embeddings, embed_fn=_embed_fn)
    else:
        progress("embed", f"Embedding {len(all_chunks)} chunks on {device}...")
        chunk_embeddings = embedder.embed_for_clustering(chunk_texts, show_progress=True, batch_size=8)

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if use_llm_summaries:
            progress("embed", "Generating LLM scenario summaries...")
            from core.llm_summarizer import generate_llm_summaries
            summaries = generate_llm_summaries(
                agreements, all_chunks, chunk_embeddings,
                embed_fn=_embed_fn, progress_fn=progress,
            )
        else:
            summaries = get_field_summaries(agreements, all_chunks, chunk_embeddings, embed_fn=_embed_fn)
        summary_embeddings = embedder.embed_for_clustering(summaries, batch_size=8)

    progress("embed", f"Embedded: {chunk_embeddings.shape[0]} chunks + {summary_embeddings.shape[0]} summaries")
    stage_end("embed")

    # Persist to DuckDB
    stage_start("persist_db")
    for i, chunk in enumerate(all_chunks):
        store.insert_chunk(
            chunk["chunk_id"], chunk["agreement_id"], chunk["text"],
            chunk["chunk_index"], chunk_embeddings[i], chunk.get("token_count", 0),
            clause_id=chunk.get("clause_id"),
        )
    for clause in all_clauses:
        store.insert_clause(
            clause["clause_id"], clause["agreement_id"],
            clause.get("clause_number", ""), clause.get("clause_title", ""),
            clause.get("heading_level", 0), clause.get("clause_index", 0),
            clause.get("full_text", ""), clause.get("token_count", 0),
        )
    stage_end("persist_db")

    # Persist to LanceDB
    stage_start("persist_lance")
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
    _save_embeddings(run_dir, chunk_embeddings, summary_embeddings)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 4: Clustering (EVoC)
    # ══════════════════════════════════════════════════════════════════════
    total_clusters = 0
    n_domains = 0
    n_outliers = 0
    n_clauses_total = len(all_clauses) if use_semantic_chunking else 0

    if mode == "macro-micro":
        # ════════════════════════════════════════════════════════════════
        # Mode A: Macro-Micro (EVoC replaces UMAP+HDBSCAN at both levels)
        # ════════════════════════════════════════════════════════════════
        stage_start("macro")
        n_docs = len(summaries)
        progress("macro", f"EVoC macro clustering {n_docs} documents...")

        if n_docs < 10:
            macro_labels = np.zeros(n_docs, dtype=int)
            domain_map = {}
            domain_id = f"d_{uuid.uuid4().hex[:8]}"
            centroid = summary_embeddings.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            domain_map[0] = {
                "domain_id": domain_id, "topic_id": 0,
                "label": "All Documents", "description": "Single domain (too few docs to cluster)",
                "agreement_count": n_docs, "centroid": centroid,
                "agreement_indices": list(range(n_docs)),
            }
            store.upsert_domain(domain_id, "All Documents", "Single domain", n_docs,
                                float(np.mean([np.dot(summary_embeddings[i], centroid) for i in range(n_docs)])), centroid)
            for i in range(n_docs):
                store.update_agreement_domain(agreements[i]["agreement_id"], domain_id,
                                              float(np.dot(summary_embeddings[i], centroid)))
            n_domains = 1
            n_outliers = 0
        else:
            macro_labels, macro_layers, macro_persistence = evoc_cluster(
                summary_embeddings,
                base_min_cluster_size=max(3, n_docs // 50),
                noise_level=evoc_noise_level,
                n_neighbors=min(15, n_docs - 1),
            )
            # Select layer preferring more domains
            macro_labels, macro_sel_idx, macro_lscores = _select_best_layer(
                macro_layers, summary_embeddings, macro_persistence,
                prefer_granularity=True,
            )
            progress("macro", f"EVoC returned {len(macro_layers)} layers, "
                     f"selected: {len(set(macro_labels) - {-1})} domains")

            # Persist all macro layers
            agr_ids = [a["agreement_id"] for a in agreements]
            store.save_cluster_layers(
                "macro", macro_layers, agr_ids,
                macro_persistence, macro_sel_idx, macro_lscores,
            )

            # LLM-label macro domains
            llm_macro_labels = _llm_label_clusters(macro_labels, summaries, "legal agreement summaries")

            unique_macro = sorted(set(t for t in macro_labels if t != -1))
            domain_map = {}

            for topic_id in unique_macro:
                mask = [i for i, t in enumerate(macro_labels) if t == topic_id]
                centroid = summary_embeddings[mask].mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

                if topic_id in llm_macro_labels:
                    info = llm_macro_labels[topic_id]
                else:
                    info = {"label": f"Domain_{topic_id}", "description": ""}

                domain_id = f"d_{uuid.uuid4().hex[:8]}"
                domain_map[topic_id] = {
                    "domain_id": domain_id, "topic_id": topic_id,
                    "label": info["label"], "description": info.get("description", ""),
                    "agreement_count": len(mask), "centroid": centroid,
                    "agreement_indices": mask,
                }
                store.upsert_domain(domain_id, info["label"], info.get("description", ""),
                                    len(mask),
                                    float(np.mean([np.dot(summary_embeddings[i], centroid) for i in mask])),
                                    centroid)
                for idx in mask:
                    conf = float(np.dot(summary_embeddings[idx], centroid))
                    store.update_agreement_domain(agreements[idx]["agreement_id"], domain_id, conf)

            # Handle macro outliers
            outlier_mask = [i for i, t in enumerate(macro_labels) if t == -1]
            if outlier_mask:
                outlier_domain_id = f"d_{uuid.uuid4().hex[:8]}"
                store.upsert_domain(outlier_domain_id, "Uncategorized", "Unassigned", len(outlier_mask))
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

        # ── Micro clustering per domain (EVoC on chunk embeddings) ──
        stage_start("micro")
        progress("micro", "EVoC micro-clustering within each domain...")
        merger = HierarchicalTopicMerger()
        quality_report = {"domains": {}, "global": {}}

        for topic_id, dr in domain_map.items():
            if topic_id == -1:
                continue
            domain_id = dr["domain_id"]
            domain_label = dr["label"]
            domain_agr_ids = set(agreements[i]["agreement_id"] for i in dr["agreement_indices"])
            domain_chunk_idx = [i for i, c in enumerate(all_chunks) if c["agreement_id"] in domain_agr_ids]

            if len(domain_chunk_idx) < 5:
                progress("micro", f"  {domain_label}: skipped ({len(domain_chunk_idx)} chunks)")
                continue

            d_texts = [all_chunks[i]["text"] for i in domain_chunk_idx]
            d_ids = [all_chunks[i]["chunk_id"] for i in domain_chunk_idx]
            d_embs = chunk_embeddings[domain_chunk_idx]
            n = len(d_texts)

            progress("micro", f"  {domain_label}: EVoC clustering {n} chunks...")
            micro_labels, micro_layers = evoc_cluster_with_target(
                d_embs,
                noise_level=evoc_noise_level,
                n_neighbors=min(15, n - 1),
            )

            # LLM-label micro clusters
            llm_micro_labels = _llm_label_clusters(micro_labels, d_texts, f'"{domain_label}" legal clauses')

            quality = compute_cluster_quality(d_embs, list(micro_labels))
            unique_micro = sorted(set(t for t in micro_labels if t != -1))

            cluster_records = []
            for tid in unique_micro:
                mask = [i for i, t in enumerate(micro_labels) if t == tid]
                c_embs = d_embs[mask]
                centroid = c_embs.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

                info = llm_micro_labels.get(tid, {"label": f"Cluster_{tid}", "description": ""})
                cluster_id = f"c_{uuid.uuid4().hex[:8]}"

                cluster_records.append({
                    "cluster_id": cluster_id, "topic_id": tid,
                    "domain_id": domain_id, "label": info["label"],
                    "description": info.get("description", ""),
                    "keywords": [],
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

            n_micro = len(cluster_records)
            total_clusters += n_micro
            progress("micro", f"  {domain_label}: {n_micro} clusters from {n} chunks")

            n_outlier_chunks = sum(1 for t in micro_labels if t == -1)
            domain_quality_scores = [q for q in quality.values() if q > 0]
            quality_report["domains"][domain_id] = {
                "label": domain_label, "n_chunks": n,
                "n_clusters": n_micro,
                "n_outlier_chunks": n_outlier_chunks,
                "avg_quality": round(float(np.mean(domain_quality_scores)), 4) if domain_quality_scores else 0,
                "clusters": [{"cluster_id": cr["cluster_id"], "label": cr["label"],
                              "chunk_count": cr["chunk_count"],
                              "quality_score": round(cr.get("quality_score", 0), 4)}
                             for cr in cluster_records],
            }

        stage_end("micro")

        all_domain_avgs = [d["avg_quality"] for d in quality_report["domains"].values() if d["avg_quality"] > 0]
        quality_report["global"] = {
            "n_documents": len(agreements), "n_chunks": len(all_chunks),
            "n_domains": n_domains, "n_clusters": total_clusters,
            "n_outlier_docs": n_outliers,
            "avg_quality_across_domains": round(float(np.mean(all_domain_avgs)), 4) if all_domain_avgs else 0,
            "clustering_algorithm": "EVoC",
            "evoc_noise_level": evoc_noise_level,
        }
        _save_quality_report(run_dir, quality_report)
        store.update_run(run_id, "micro_complete")

    elif mode in ("clause", "hybrid"):
        # ════════════════════════════════════════════════════════════════
        # Mode B/C: Clause-type clustering (EVoC on clause embeddings)
        # Hybrid mode adds macro document-domain clustering on top.
        # ════════════════════════════════════════════════════════════════
        stage_start("clause_cluster")

        # Compute clause-level embeddings
        clause_id_to_chunk_indices = {}
        for i, c in enumerate(all_chunks):
            cid = c.get("clause_id")
            if cid:
                clause_id_to_chunk_indices.setdefault(cid, []).append(i)

        valid_clauses = [cl for cl in all_clauses if cl["clause_id"] in clause_id_to_chunk_indices]
        n_clauses = len(valid_clauses)
        progress("clause_cluster", f"Computing clause embeddings for {n_clauses} clauses...")

        clause_embeddings = np.zeros((n_clauses, chunk_embeddings.shape[1]), dtype=np.float32)
        for ci, cl in enumerate(valid_clauses):
            idx = clause_id_to_chunk_indices[cl["clause_id"]]
            emb = chunk_embeddings[idx].mean(axis=0)
            norm = np.linalg.norm(emb)
            clause_embeddings[ci] = emb / norm if norm > 0 else emb

        clause_texts = []
        for cl in valid_clauses:
            title = cl.get("clause_title", "")
            body = cl.get("full_text", "")[:1500]
            clause_texts.append(f"{title}\n{body}" if title else body)

        # EVoC clause clustering — with or without Optuna
        if n_clauses < 10:
            clause_topics = np.zeros(n_clauses, dtype=int)
            optuna_best_params = None
        elif use_optuna and n_clauses >= 20:
            # Optuna optimization: tunes EVoC params + layer selection
            progress("clause_cluster", f"Optuna optimizing EVoC ({optuna_trials} trials, "
                     f"{n_clauses} clauses)...")
            optuna_best_params, clause_topics, optuna_score = evoc_optimize(
                clause_embeddings,
                n_trials=optuna_trials,
                min_clusters=max(3, n_clauses // 200),
                progress_fn=progress,
            )
            n_types = len(set(clause_topics) - {-1})
            n_out = sum(1 for t in clause_topics if t == -1)
            progress("clause_cluster", f"Optuna best: {n_types} types, {n_out} outliers "
                     f"({n_out/n_clauses*100:.1f}%), score={optuna_score:.4f}")
            progress("clause_cluster", f"Optuna params: {optuna_best_params}")
        else:
            # Direct EVoC with tuned defaults
            progress("clause_cluster", f"EVoC clustering {n_clauses} clauses "
                     f"(noise_level={evoc_noise_level})...")
            clause_topics, clause_layers, clause_persistence = evoc_cluster(
                clause_embeddings,
                base_min_cluster_size=max(3, n_clauses // 100),
                noise_level=evoc_noise_level,
                n_neighbors=min(15, n_clauses - 1),
                min_samples=3,
            )

            # Use scoring-based layer selection instead of persistence
            clause_topics, best_layer_idx, clause_lscores = _select_best_layer(
                clause_layers, clause_embeddings, clause_persistence,
            )
            n_types = len(set(clause_topics) - {-1})
            n_out = sum(1 for t in clause_topics if t == -1)
            progress("clause_cluster", f"Selected layer {best_layer_idx}: {n_types} types, "
                     f"{n_out} outliers ({n_out/n_clauses*100:.1f}%)")
            optuna_best_params = None

            # Log all layers
            for li, layer in enumerate(clause_layers):
                lt = len(set(layer) - {-1})
                lo = sum(1 for t in layer if t == -1)
                progress("clause_cluster", f"  Layer {li}: {lt} types, {lo} outliers "
                         f"(persistence={clause_persistence[li]:.4f})"
                         f"{' ← SELECTED' if li == best_layer_idx else ''}")

            # Persist all clause layers
            clause_item_ids = [cl["clause_id"] for cl in valid_clauses]
            store.save_cluster_layers(
                "clause", clause_layers, clause_item_ids,
                clause_persistence, best_layer_idx, clause_lscores,
            )

        # LLM-label clause types
        llm_clause_labels = _llm_label_clusters(clause_topics, clause_texts, "legal clause sections")

        # Build cluster records
        unique_clause_types = sorted(set(t for t in clause_topics if t != -1))
        quality = compute_cluster_quality(clause_embeddings, list(clause_topics))

        cluster_records = []
        all_domain_id = f"d_{uuid.uuid4().hex[:8]}"
        for tid in unique_clause_types:
            mask = [i for i, t in enumerate(clause_topics) if t == tid]
            c_embs = clause_embeddings[mask]
            centroid = c_embs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroid = centroid / norm if norm > 0 else centroid

            info = llm_clause_labels.get(tid, {"label": f"Clause_Type_{tid}", "description": ""})
            cluster_id = f"ct_{uuid.uuid4().hex[:8]}"

            type_chunk_ids = []
            type_clause_ids = []
            type_agr_ids = set()
            for ci in mask:
                cl = valid_clauses[ci]
                type_clause_ids.append(cl["clause_id"])
                type_chunk_ids.extend(cl.get("chunk_ids", []))
                type_agr_ids.add(cl["agreement_id"])

            cluster_records.append({
                "cluster_id": cluster_id, "topic_id": tid,
                "domain_id": all_domain_id, "label": info["label"],
                "description": info.get("description", ""),
                "keywords": [],
                "chunk_count": len(type_chunk_ids),
                "clause_count": len(mask),
                "agreement_count": len(type_agr_ids),
                "quality_score": quality.get(tid, 0.0),
                "centroid": centroid,
                "chunk_ids": type_chunk_ids,
                "clause_ids": type_clause_ids,
            })

        # Merge near-duplicate clause types (raised threshold for EVoC — diagnostics
        # showed 0.93 was too aggressive, collapsing types improperly)
        if len(cluster_records) > 1:
            merger = HierarchicalTopicMerger(merge_threshold=0.96)
            pre_merge = len(cluster_records)
            cluster_records = merger.merge_similar_topics(cluster_records)
            if len(cluster_records) < pre_merge:
                progress("clause_cluster", f"Merged {pre_merge} → {len(cluster_records)} clause types")

        total_clusters = len(cluster_records)
        n_domains = 1
        n_outliers = 0

        # Persist clause types
        store.upsert_domain(all_domain_id, "All Documents", "EVoC clause-type clustered corpus",
                            len(agreements), 1.0)
        for agr in agreements:
            store.update_agreement_domain(agr["agreement_id"], all_domain_id, 1.0)

        for cr in cluster_records:
            store.upsert_cluster(
                cr["cluster_id"], all_domain_id, cr["label"], cr.get("description", ""),
                cr.get("keywords", []), cr["chunk_count"], cr.get("agreement_count", 0),
                cr.get("quality_score"), cr.get("centroid"),
                cr.get("aliases", []), cr.get("merged_from", []),
            )
            if cr.get("centroid") is not None:
                store.save_centroid(cr["cluster_id"], cr["centroid"], cr["chunk_count"])
            for cid in cr.get("chunk_ids", []):
                store.insert_cluster_assignment(cid, cr["cluster_id"])
            for cls_id in cr.get("clause_ids", []):
                store.update_clause_type(cls_id, cr["cluster_id"])

        # Keep outlier clauses as outliers — assign to nearest cluster
        # with is_outlier=True so they're visible but don't pollute clusters
        outlier_clause_indices = [i for i, t in enumerate(clause_topics) if t == -1]
        if outlier_clause_indices and cluster_records:
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            type_centroids = np.array([cr["centroid"] for cr in cluster_records])
            for oi in outlier_clause_indices:
                cl = valid_clauses[oi]
                sims = cos_sim(clause_embeddings[oi:oi+1], type_centroids)[0]
                best = int(np.argmax(sims))
                best_sim = float(sims[best])
                best_cr = cluster_records[best]
                for cid in cl.get("chunk_ids", []):
                    store.insert_cluster_assignment(
                        cid, best_cr["cluster_id"],
                        probability=best_sim, is_outlier=True,
                    )
                # Do NOT update clause_type — leave it as None/unassigned
            progress("clause_cluster",
                     f"Marked {len(outlier_clause_indices)} outlier clauses "
                     f"(nearest-cluster attached with is_outlier=True)")

        # Quality report for clause types
        quality_report = {"domains": {}, "global": {}}
        all_quality_scores = [q for q in quality.values() if q > 0]
        n_outlier_clauses = sum(1 for t in clause_topics if t == -1)
        quality_report["global"] = {
            "n_documents": len(agreements), "n_chunks": len(all_chunks),
            "n_clauses": n_clauses, "n_clause_types": total_clusters,
            "n_outlier_clauses": n_outlier_clauses,
            "clause_outlier_ratio": round(n_outlier_clauses / n_clauses, 4) if n_clauses > 0 else 0,
            "avg_quality": round(float(np.mean(all_quality_scores)), 4) if all_quality_scores else 0,
            "clustering_algorithm": "EVoC",
            "evoc_noise_level": evoc_noise_level,
            "use_optuna": use_optuna,
            "optuna_trials": optuna_trials if use_optuna else 0,
            "optuna_best_params": optuna_best_params,
        }
        quality_report["clause_types"] = [
            {"cluster_id": cr["cluster_id"], "label": cr["label"],
             "clause_count": cr.get("clause_count", 0),
             "chunk_count": cr["chunk_count"],
             "quality_score": round(cr.get("quality_score", 0), 4)}
            for cr in cluster_records
        ]

        progress("clause_cluster", f"{total_clusters} clause types, "
                 f"avg_quality={quality_report['global']['avg_quality']:.4f}")
        stage_end("clause_cluster")
        store.update_run(run_id, "clause_cluster_complete")
        n_clauses_total = n_clauses

        # ════════════════════════════════════════════════════════════════
        # Hybrid extension: Macro document-domain clustering
        # ════════════════════════════════════════════════════════════════
        if mode == "hybrid":
            stage_start("macro")
            n_docs = len(summaries)
            progress("macro", f"Hybrid macro: EVoC clustering {n_docs} documents into domains...")

            if n_docs < 10:
                macro_labels = np.zeros(n_docs, dtype=int)
                macro_domain_map = {}
                domain_id = f"d_{uuid.uuid4().hex[:8]}"
                centroid = summary_embeddings.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
                macro_domain_map[0] = {
                    "domain_id": domain_id, "topic_id": 0,
                    "label": "All Documents", "description": "Single domain (too few docs)",
                    "agreement_count": n_docs, "centroid": centroid,
                    "agreement_indices": list(range(n_docs)),
                }
                store.upsert_domain(domain_id, "All Documents", "Single domain", n_docs,
                                    float(np.mean([np.dot(summary_embeddings[i], centroid) for i in range(n_docs)])),
                                    centroid)
                for i in range(n_docs):
                    store.update_agreement_domain(agreements[i]["agreement_id"], domain_id,
                                                  float(np.dot(summary_embeddings[i], centroid)))
                n_domains = 1
            else:
                macro_labels, macro_layers, macro_persistence = evoc_cluster(
                    summary_embeddings,
                    base_min_cluster_size=max(3, n_docs // 50),
                    noise_level=evoc_noise_level,
                    n_neighbors=min(15, n_docs - 1),
                    min_samples=2,
                )
                # Layer selection for macro — prefer more granular domains
                macro_labels, macro_layer_idx, macro_lscores = _select_best_layer(
                    macro_layers, summary_embeddings, macro_persistence,
                    prefer_granularity=True,
                )

                # Persist all macro layers
                agr_ids_list = [a["agreement_id"] for a in agreements]
                store.save_cluster_layers(
                    "macro", macro_layers, agr_ids_list,
                    macro_persistence, macro_layer_idx, macro_lscores,
                )

                llm_macro_labels = _llm_label_clusters(
                    macro_labels, summaries, "legal agreement summaries",
                )

                unique_macro = sorted(set(t for t in macro_labels if t != -1))
                macro_domain_map = {}

                for topic_id in unique_macro:
                    mask = [i for i, t in enumerate(macro_labels) if t == topic_id]
                    centroid = summary_embeddings[mask].mean(axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

                    info = llm_macro_labels.get(topic_id,
                                                {"label": f"Domain_{topic_id}", "description": ""})
                    domain_id = f"d_{uuid.uuid4().hex[:8]}"
                    macro_domain_map[topic_id] = {
                        "domain_id": domain_id, "topic_id": topic_id,
                        "label": info["label"],
                        "description": info.get("description", ""),
                        "agreement_count": len(mask), "centroid": centroid,
                        "agreement_indices": mask,
                    }
                    store.upsert_domain(domain_id, info["label"], info.get("description", ""),
                                        len(mask),
                                        float(np.mean([np.dot(summary_embeddings[i], centroid)
                                                       for i in mask])),
                                        centroid)
                    for idx in mask:
                        conf = float(np.dot(summary_embeddings[idx], centroid))
                        store.update_agreement_domain(agreements[idx]["agreement_id"],
                                                      domain_id, conf)

                # Outlier docs → nearest domain
                outlier_docs = [i for i, t in enumerate(macro_labels) if t == -1]
                if outlier_docs and macro_domain_map:
                    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
                    domain_centroids = np.array([d["centroid"] for d in macro_domain_map.values()])
                    domain_ids = [d["domain_id"] for d in macro_domain_map.values()]
                    for oi in outlier_docs:
                        sims = cos_sim(summary_embeddings[oi:oi+1], domain_centroids)[0]
                        best_d = domain_ids[int(np.argmax(sims))]
                        store.update_agreement_domain(agreements[oi]["agreement_id"],
                                                      best_d, float(np.max(sims)))

                n_domains = len(unique_macro)
                n_outliers = len(outlier_docs)

            progress("macro", f"Hybrid domains: {n_domains} document domains, "
                     f"{n_outliers} outlier docs")
            stage_end("macro")

            # Add domain info to quality report
            quality_report["global"]["n_domains"] = n_domains
            quality_report["global"]["n_outlier_docs"] = n_outliers
            quality_report["global"]["mode"] = "hybrid"
            quality_report["domains"] = {
                d["domain_id"]: {
                    "label": d["label"],
                    "agreement_count": d["agreement_count"],
                }
                for d in macro_domain_map.values()
            }

        _save_quality_report(run_dir, quality_report)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'clause', 'macro-micro', or 'hybrid'.")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 5: Field Discovery (Azure OpenAI)
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
            if use_rlm:
                progress("field_discovery", "Using dspy.RLM for field discovery (recursive exploration)")
            field_discovery_result = run_field_discovery(
                store=store, config=fd_config,
                progress_fn=progress, extract_values=True,
                min_cluster_chunks=5,
                max_extraction_workers=5,
                min_confidence=0.5,
                use_rlm=use_rlm,
            )
        except Exception as e:
            logger.error(f"Field discovery failed: {e}")
            progress("field_discovery", f"Failed: {e}")
        stage_end("field_discovery")

        # Universal field dedup
        try:
            _dedup_universal_fields(store, progress)
        except Exception as e:
            logger.warning(f"Field dedup failed (non-fatal): {e}")
    else:
        progress("field_discovery", "Skipped (AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY not set)")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 6: Intent Extraction (Azure OpenAI)
    # ══════════════════════════════════════════════════════════════════════
    intent_result = {"total_extracted": 0, "total_failed": 0}
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        stage_start("intent")
        progress("intent", "Extracting clause intents via Azure OpenAI...")
        try:
            from core.intent_extractor import run_intent_extraction, IntentExtractionConfig
            intent_config = IntentExtractionConfig(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4"),
            )
            intent_result = run_intent_extraction(
                store=store, config=intent_config,
                progress_fn=progress,
            )
        except Exception as e:
            logger.error(f"Intent extraction failed: {e}")
            progress("intent", f"Failed: {e}")
        stage_end("intent")
    else:
        progress("intent", "Skipped (AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY not set)")

    store.update_run(run_id, "completed", "completed")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 7: Knowledge Graph Analysis
    # ══════════════════════════════════════════════════════════════════════
    kg_result = {}
    stage_start("knowledge_graph")
    progress("knowledge_graph", "Building knowledge graph...")
    try:
        from core.knowledge_graph import ContractKnowledgeGraph
        kg = ContractKnowledgeGraph(store)
        kg_stats = kg.build()
        progress("knowledge_graph",
                 f"Graph: {kg_stats.agreements} agreements, "
                 f"{kg_stats.clause_types} clause types, "
                 f"{kg_stats.intent_types} intent types, "
                 f"{kg_stats.edges} edges")

        # Add embedding similarity if intents exist
        try:
            kg.add_intent_similarity(embedder)
            sim_edges = sum(1 for _, _, d in kg.G.edges(data=True)
                           if d.get("rel") == "SIMILAR_INTENT")
            if sim_edges:
                progress("knowledge_graph",
                         f"Added {sim_edges} intent similarity edges")
        except Exception as e:
            logger.warning(f"Intent similarity skipped: {e}")

        rules = kg.get_implication_rules(min_confidence=0.6, min_support=3)
        strategies = kg.discover_strategies(min_cluster_size=2)
        archetypes = kg.discover_archetypes(min_agreements=2)

        kg_result = {
            "nodes": kg.G.number_of_nodes(),
            "edges": kg.G.number_of_edges(),
            "clause_types": kg_stats.clause_types,
            "intent_types": kg_stats.intent_types,
            "parties": kg_stats.parties,
            "implication_rules": len(rules),
            "strategies": len(strategies),
            "archetypes": len(archetypes),
        }

        # Export graph artifacts
        kg_dir = run_dir / "knowledge_graph"
        kg_dir.mkdir(exist_ok=True)
        import json as _json
        (kg_dir / "implication_rules.json").write_text(
            _json.dumps(rules[:50], indent=2))
        (kg_dir / "strategies.json").write_text(
            _json.dumps(strategies, indent=2))
        (kg_dir / "archetypes.json").write_text(
            _json.dumps(archetypes, indent=2))

        # Per-agreement recommendations, anomalies, and similar agreements
        agr_insights = {}
        for agr_id in kg._agr_clause_types:
            recs = kg.recommend(agr_id, top_n=5)
            anomalies = kg.detect_anomalies(agr_id)
            similar = kg.find_most_similar_agreements(agr_id, top_n=5)
            fingerprint = kg.get_agreement_fingerprint(agr_id)
            agr_insights[agr_id] = {
                "clause_types": fingerprint["clause_types"],
                "n_clause_types": fingerprint["n_clause_types"],
                "n_intents": fingerprint["n_intents"],
                "similar_agreements": similar,
                "recommendations": recs,
                "anomalies": anomalies,
            }
        (kg_dir / "agreement_insights.json").write_text(
            _json.dumps(agr_insights, indent=2))

        # Agreement similarity network (edges with Jaccard ≥ 0.3)
        agr_network = []
        agr_list = list(kg._agr_clause_types.keys())
        for i, a1 in enumerate(agr_list):
            s1 = kg._agr_clause_types[a1]
            for a2 in agr_list[i+1:]:
                s2 = kg._agr_clause_types[a2]
                shared = s1 & s2
                if not shared:
                    continue
                jaccard = len(shared) / len(s1 | s2)
                if jaccard >= 0.2:
                    agr_network.append({
                        "source": a1, "target": a2,
                        "jaccard": round(jaccard, 3),
                        "shared_types": len(shared),
                        "shared_labels": sorted(
                            kg.G.nodes[ct].get("label", ct) for ct in shared),
                    })
        agr_network.sort(key=lambda e: -e["jaccard"])
        (kg_dir / "agreement_network.json").write_text(
            _json.dumps(agr_network, indent=2))

        progress("knowledge_graph",
                 f"Done: {len(rules)} rules, {len(strategies)} strategies, "
                 f"{len(archetypes)} archetypes, "
                 f"{len(agr_insights)} agreements with insights, "
                 f"{len(agr_network)} agreement-pair connections")
    except Exception as e:
        logger.error(f"Knowledge graph failed: {e}")
        progress("knowledge_graph", f"Failed: {e}")
    stage_end("knowledge_graph")

    # ══════════════════════════════════════════════════════════════════════
    # Export JSON artifacts
    # ══════════════════════════════════════════════════════════════════════
    progress("export", "Exporting JSON artifacts...")
    _export_fields_json(store, run_dir)
    _export_chunks_json(store, run_dir)
    if mode in ("clause", "hybrid"):
        _export_clause_types_json(store, run_dir, progress)

    # ══════════════════════════════════════════════════════════════════════
    # Save manifest
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    stats = store.get_stats()
    manifest = {
        "run_id": run_id,
        "account_id": account_id,
        "pipeline": "evoc",
        "mode": mode,
        "clustering_algorithm": "EVoC",
        "evoc_noise_level": evoc_noise_level,
        "pdf_dir": str(pdf_dir),
        "n_documents": len(agreements),
        "n_chunks": len(all_chunks),
        "n_clauses": n_clauses_total,
        "n_clause_types": total_clusters if mode in ("clause", "hybrid") else 0,
        "n_domains": n_domains,
        "n_clusters": total_clusters,
        "n_outlier_docs": n_outliers,
        "embedding_dim": int(embedder.dim),
        "device": device,
        "cuml_available": cuml_available(),
        "use_semantic_chunking": use_semantic_chunking,
        "use_optuna": use_optuna,
        "optuna_trials": optuna_trials if use_optuna else 0,
        "elapsed_seconds": round(elapsed, 1),
        "stage_timings": stage_timings,
        "db_path": str(db),
        "lance_path": str(lance),
        "artifacts_dir": str(run_dir),
        "embeddings_path": str(run_dir / "embeddings.npz"),
        "quality_report_path": str(run_dir / "cluster_quality.json"),
        "fields_path": str(run_dir / "fields.json"),
        "chunks_path": str(run_dir / "chunks.json"),
        "field_discovery": field_discovery_result,
        "intent_extraction": intent_result,
        "knowledge_graph": kg_result,
        "stats": stats,
    }
    if mode in ("clause", "hybrid"):
        manifest["clause_types_path"] = str(run_dir / "clause_types.json")
    _save_manifest(run_dir, manifest)

    mode_label = 'clause types' if mode in ('clause', 'hybrid') else 'clusters'
    progress("done", f"Complete in {elapsed:.0f}s — {len(agreements)} docs, "
             f"{len(all_chunks)} chunks, "
             f"{n_clauses_total} clauses, "
             f"{total_clusters} {mode_label}"
             f"{f', {n_domains} domains' if mode == 'hybrid' else ''}")
    progress("done", f"Artifacts: {run_dir}")

    return {
        "store": store,
        "lance_store": lance_store,
        "run_id": run_id,
        "account_id": account_id,
        "stats": stats,
        "n_documents": len(agreements),
        "n_chunks": len(all_chunks),
        "n_clauses": n_clauses_total,
        "n_clause_types": total_clusters if mode in ("clause", "hybrid") else 0,
        "n_domains": n_domains,
        "n_clusters": total_clusters,
        "elapsed_seconds": round(elapsed, 1),
        "artifacts_dir": str(run_dir),
        "manifest": manifest,
    }


# ── Post-Pipeline Analytics ────────────────────────────────────────────────────
def run_analytics(db_path: str, use_llm_risk: bool = False, min_instances: int = 10):
    """Run all 4 analytics modules on a DuckDB database.
    
    Can be called standalone or chained after run_evoc_pipeline().
    """
    import time as _t

    # 1. Risk Scoring
    logger.info("Analytics: Risk scoring...")
    t0 = _t.time()
    from core.risk_scorer import RiskScorer
    llm = None
    if use_llm_risk:
        from core.llm_client import LLMClient
        llm = LLMClient()
    scorer = RiskScorer(db_path, llm=llm)
    scorer.score_all_clusters(use_llm=use_llm_risk)
    scorer.score_all_agreements()
    print(scorer.summary())
    scorer.close()
    logger.info(f"Analytics: Risk scoring done ({_t.time()-t0:.1f}s)")

    # 2. Clause Deviation
    logger.info("Analytics: Clause deviation scoring...")
    t0 = _t.time()
    from core.clause_deviation import ClauseDeviationScorer
    dev = ClauseDeviationScorer(db_path, min_instances=min_instances)
    dev.score_all()
    print(dev.summary())
    dev.close()
    logger.info(f"Analytics: Deviation scoring done ({_t.time()-t0:.1f}s)")

    # 3. Field Normalization
    logger.info("Analytics: Field normalization...")
    t0 = _t.time()
    from core.field_normalizer import FieldNormalizer
    norm = FieldNormalizer(db_path)
    norm.normalize_all()
    print(norm.summary())
    norm.close()
    logger.info(f"Analytics: Normalization done ({_t.time()-t0:.1f}s)")

    # 4. Playbook / Benchmark Library
    logger.info("Analytics: Playbook benchmarks...")
    t0 = _t.time()
    from core.playbook import PlaybookLibrary
    pb = PlaybookLibrary(db_path)
    pb.build_benchmarks(min_instances=min_instances)
    print(pb.summary())
    pb.close()
    logger.info(f"Analytics: Playbook done ({_t.time()-t0:.1f}s)")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Run EVoC clustering pipeline on PDFs")
    parser.add_argument("pdf_dir", help="Directory containing PDF files")
    parser.add_argument("--account-id", default="default", help="Account/tenant ID")
    parser.add_argument("--mode", choices=["clause", "macro-micro", "hybrid"], default="hybrid",
                        help="Clustering mode: 'clause', 'macro-micro', or 'hybrid' (recommended)")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--fast-chunk", action="store_true", help="Use sliding window instead of semantic chunking")
    parser.add_argument("--llm-summaries", action="store_true",
                        help="Use Azure OpenAI to generate scenario summaries")
    parser.add_argument("--noise-level", type=float, default=0.3,
                        help="EVoC noise_level (0.0=cluster more, 1.0=purer clusters)")
    parser.add_argument("--no-optuna", action="store_true",
                        help="Disable Optuna optimization (use fixed defaults)")
    parser.add_argument("--optuna-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--refine", action="store_true",
                        help="Run ontology refinement (outlier assign, field dedup, label synthesis)")
    parser.add_argument("--analytics", action="store_true",
                        help="Run post-pipeline analytics (risk, deviation, normalization, playbook)")
    parser.add_argument("--use-llm-risk", action="store_true",
                        help="Use LLM for risk scoring (default: keyword heuristic)")
    parser.add_argument("--use-rlm", action="store_true",
                        help="Use dspy.RLM for field discovery (explores all chunks programmatically)")
    parser.add_argument("--min-instances", type=int, default=10,
                        help="Min clauses per cluster for deviation/playbook scoring (default: 10)")
    args = parser.parse_args()

    result = run_evoc_pipeline(
        pdf_dir=args.pdf_dir,
        account_id=args.account_id,
        mode=args.mode,
        max_docs=args.max_docs,
        device=args.device,
        use_semantic_chunking=not args.fast_chunk,
        use_llm_summaries=args.llm_summaries,
        evoc_noise_level=args.noise_level,
        use_optuna=not args.no_optuna,
        optuna_trials=args.optuna_trials,
        use_rlm=args.use_rlm,
    )

    print(f"\n{'='*60}")
    print(f"  Pipeline:     EVoC ({args.mode})")
    print(f"  Optuna:       {'yes' if not args.no_optuna else 'no'} ({args.optuna_trials} trials)")
    print(f"  Documents:    {result['n_documents']}")
    print(f"  Chunks:       {result['n_chunks']}")
    if args.mode in ("clause", "hybrid"):
        print(f"  Clauses:      {result.get('n_clauses', 0)}")
        print(f"  Clause Types: {result.get('n_clause_types', 0)}")
    if args.mode in ("macro-micro", "hybrid"):
        print(f"  Domains:      {result['n_domains']}")
    if args.mode == "macro-micro":
        print(f"  Clusters:     {result['n_clusters']}")
    print(f"  Time:         {result['elapsed_seconds']}s")
    print(f"  Artifacts:    {result['artifacts_dir']}")
    print(f"{'='*60}")

    # Determine db_path for post-pipeline stages
    _db_path = str(DATA_DIR / f"{args.account_id}.duckdb")

    # ── Ontology Refinement ──
    if args.refine:
        print(f"\n{'='*60}")
        print("  Running ontology refinement...")
        print(f"{'='*60}")
        from core.ontology_refiner import OntologyRefiner
        refiner = OntologyRefiner(_db_path)
        refiner.assign_outliers()
        refiner.deduplicate_fields()
        try:
            refiner.synthesize_labels()
        except Exception as e:
            logger.warning(f"Label synthesis skipped: {e}")
        refiner.close()
        # Update db_path to refined version if it was created
        refined_path = _db_path.replace(".duckdb", "_refined.duckdb")
        import os
        if os.path.exists(refined_path):
            _db_path = refined_path

    # ── Post-Pipeline Analytics ──
    if args.analytics:
        print(f"\n{'='*60}")
        print("  Running post-pipeline analytics...")
        print(f"{'='*60}")
        run_analytics(_db_path,
                      use_llm_risk=args.use_llm_risk,
                      min_instances=args.min_instances)
