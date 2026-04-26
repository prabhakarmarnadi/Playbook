"""
Production Ensemble Backend
============================

Combines all four discovery methods under a single API with an ensemble
strategy that maximizes coverage while minimizing hallucination.

Ensemble Strategy:
    1. DISCOVER — Run RLM v2 (broadest coverage) + Standard GPT (fast baseline)
    2. MERGE    — Union of fields, deduplicated by embedding similarity
    3. EXTRACT  — Single extraction pass with the merged field set
    4. VERIFY   — LLM judge filters hallucinated values (optional, adds latency)
    5. STORE    — Corrections feed back as gold labels for precision/recall tracking

Endpoints:
    POST /api/v1/discover         Field discovery for a cluster (ensemble)
    POST /api/v1/extract          Extract fields from text
    POST /api/v1/discover-extract Full pipeline: discover + extract in one call
    POST /api/v1/judge            Judge extraction quality (LLM-as-judge)
    POST /api/v1/correct          Submit human correction (builds gold set)

    GET  /api/v1/clusters                 List clusters
    GET  /api/v1/clusters/{id}            Cluster detail + fields
    GET  /api/v1/clusters/{id}/fields     Field definitions for cluster
    GET  /api/v1/agreements/{id}          Agreement with extractions
    GET  /api/v1/search                   Semantic search across chunks

    GET  /api/v1/eval/metrics             Run eval suite (precision/recall if gold exists)
    GET  /api/v1/eval/compare             Method comparison summary

    POST /api/v1/pipeline/run             Full pipeline (background job)
    GET  /api/v1/jobs/{job_id}            Job status
    GET  /api/v1/health                   Health check

Usage:
    cd clustering-v2/hackathon
    source /home/azureuser/.venv/bin/activate
    uvicorn api_server_v2:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

load_dotenv()

from config import DATA_DIR, DB_PATH
from core.field_discovery import (
    FieldDiscoveryConfig,
    discover_fields_for_cluster,
    discover_fields_for_cluster_rlm,
    discover_fields_for_cluster_rlm_lb,
    extract_fields_from_chunk,
    _get_azure_client,
    _get_deployment,
)
from core.rlm_v2 import discover_fields_for_cluster_rlm_v2

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)


# ═══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="EVoC Extraction API",
    description="Ensemble field discovery + extraction backend for contract analysis",
    version="2.0.0",
)

# ── Shared state ──────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_embedder = None  # lazy-loaded SentenceTransformer


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_db(db_path: str | None = None, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    path = db_path or str(DB_PATH)
    return duckdb.connect(path, read_only=read_only)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class DiscoveryMethod(str, Enum):
    standard_gpt = "standard_gpt"
    dspy_rlm = "dspy_rlm"
    rlm_lb = "rlm_lb"
    rlm_v2 = "rlm_v2"
    ensemble = "ensemble"


class DiscoverRequest(BaseModel):
    cluster_id: str = Field(..., description="Cluster ID to discover fields for")
    method: DiscoveryMethod = Field(default=DiscoveryMethod.ensemble)
    db_path: str | None = Field(default=None, description="Override DB path")


class ExtractRequest(BaseModel):
    cluster_label: str = Field(..., description="Cluster/clause type label")
    text: str = Field(..., description="Text to extract from")
    fields: list[dict] = Field(..., description="Field definitions [{name, description}, ...]")
    db_path: str | None = Field(default=None)


class DiscoverExtractRequest(BaseModel):
    cluster_id: str = Field(..., description="Cluster ID")
    chunk_ids: list[str] | None = Field(default=None, description="Specific chunks (default: sample)")
    n_chunks: int = Field(default=8, description="Number of chunks to extract from")
    method: DiscoveryMethod = Field(default=DiscoveryMethod.ensemble)
    verify: bool = Field(default=False, description="Run LLM judge on results")
    db_path: str | None = Field(default=None)


class CorrectionRequest(BaseModel):
    cluster_id: str
    chunk_id: str
    field_name: str
    corrected_value: str | None = Field(description="Correct value, or null if not present")
    original_value: str | None = Field(description="What the model extracted")
    db_path: str | None = Field(default=None)


class JudgeRequest(BaseModel):
    cluster_label: str
    chunk_text: str
    extractions: list[dict] = Field(
        ..., description="[{field_name, extracted_value}, ...]"
    )


class PipelineRequest(BaseModel):
    pdf_dir: str
    account_id: str = "default"
    max_docs: int | None = None
    device: str = "cuda"


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def _deduplicate_fields(fields_lists: list[list[dict]], threshold: float = 0.82) -> list[dict]:
    """
    Merge field lists from multiple methods. Dedup by embedding similarity.
    When two fields collide, keep the one with a description (or the first seen).
    """
    if not any(fields_lists):
        return []

    embedder = _get_embedder()
    all_fields = []
    seen_names: set[str] = set()

    for fields in fields_lists:
        for f in (fields or []):
            norm_name = f.get("name", "").strip().lower()
            if norm_name and norm_name not in seen_names:
                all_fields.append(f)
                seen_names.add(norm_name)

    if len(all_fields) <= 15:
        return all_fields

    # Embedding-based dedup for near-duplicates
    names = [f["name"] for f in all_fields]
    embeddings = embedder.encode(names, normalize_embeddings=True)
    sim_matrix = embeddings @ embeddings.T

    keep = [True] * len(all_fields)
    for i in range(len(all_fields)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(all_fields)):
            if not keep[j]:
                continue
            if sim_matrix[i][j] > threshold:
                # Keep the one with a description, else keep first
                if all_fields[j].get("description") and not all_fields[i].get("description"):
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    merged = [f for f, k in zip(all_fields, keep) if k]

    # Cap at 20 fields — sort by those appearing in more methods
    if len(merged) > 20:
        merged = merged[:20]

    return merged


def discover_ensemble(
    cluster_label: str,
    chunks: list[str],
    config: FieldDiscoveryConfig,
    client,
    deployment: str,
) -> dict:
    """
    Ensemble discovery: run Standard GPT + RLM v2, merge fields.
    Returns {fields, methods_used, timings, v2_bonus}.
    """
    timings = {}

    # Method 1: Standard GPT (fast, reliable baseline)
    t0 = time.time()
    try:
        fields_std = discover_fields_for_cluster(
            client, deployment, cluster_label, chunks, config,
        )
    except Exception as e:
        logger.warning(f"Standard GPT discovery failed: {e}")
        fields_std = []
    timings["standard_gpt"] = round(time.time() - t0, 2)

    # Method 2: RLM v2 (broadest coverage + bonus outputs)
    t0 = time.time()
    v2_bonus = {}
    try:
        v2_result = discover_fields_for_cluster_rlm_v2(cluster_label, chunks, config)
        fields_v2 = v2_result["fields"]
        v2_bonus = {
            "clause_types": v2_result.get("clause_types", []),
            "intents": v2_result.get("intents", []),
            "obligations": v2_result.get("obligations", []),
        }
    except Exception as e:
        logger.warning(f"RLM v2 discovery failed: {e}")
        fields_v2 = []
    timings["rlm_v2"] = round(time.time() - t0, 2)

    # Merge
    merged = _deduplicate_fields([fields_std, fields_v2])

    return {
        "fields": merged,
        "methods_used": ["standard_gpt", "rlm_v2"],
        "timings": timings,
        "v2_bonus": v2_bonus,
        "method_fields": {
            "standard_gpt": len(fields_std),
            "rlm_v2": len(fields_v2),
            "merged": len(merged),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — DISCOVERY & EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/discover")
async def discover_fields(request: DiscoverRequest):
    """Discover fields for a cluster using the specified method."""
    conn = _get_db(request.db_path)
    config = FieldDiscoveryConfig()
    client = _get_azure_client()
    deployment = _get_deployment(config)

    # Fetch cluster info + sample chunks
    cluster = conn.execute(
        "SELECT cluster_id, label, chunk_count FROM clusters WHERE cluster_id = ?",
        [request.cluster_id],
    ).fetchone()
    if not cluster:
        raise HTTPException(404, f"Cluster not found: {request.cluster_id}")

    cluster_id, cluster_label, chunk_count = cluster

    chunks = conn.execute("""
        SELECT ch.chunk_text FROM chunks ch
        JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
        WHERE ca.cluster_id = ?
        ORDER BY RANDOM() LIMIT 30
    """, [cluster_id]).fetchdf()["chunk_text"].tolist()

    conn.close()

    t0 = time.time()
    method = request.method

    if method == DiscoveryMethod.ensemble:
        result = discover_ensemble(cluster_label, chunks, config, client, deployment)
        return {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "method": "ensemble",
            **result,
            "elapsed_s": round(time.time() - t0, 2),
        }

    # Single method
    if method == DiscoveryMethod.standard_gpt:
        fields = discover_fields_for_cluster(client, deployment, cluster_label, chunks, config)
    elif method == DiscoveryMethod.dspy_rlm:
        fields = discover_fields_for_cluster_rlm(cluster_label, chunks, config)
    elif method == DiscoveryMethod.rlm_lb:
        fields = discover_fields_for_cluster_rlm_lb(cluster_label, chunks, config)
    elif method == DiscoveryMethod.rlm_v2:
        v2_result = discover_fields_for_cluster_rlm_v2(cluster_label, chunks, config)
        return {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "method": method.value,
            "fields": v2_result["fields"],
            "v2_bonus": {
                "clause_types": v2_result.get("clause_types", []),
                "intents": v2_result.get("intents", []),
                "obligations": v2_result.get("obligations", []),
            },
            "elapsed_s": round(time.time() - t0, 2),
        }
    else:
        raise HTTPException(400, f"Unknown method: {method}")

    return {
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "method": method.value,
        "fields": fields,
        "elapsed_s": round(time.time() - t0, 2),
    }


@app.post("/api/v1/extract")
async def extract_from_text(request: ExtractRequest):
    """Extract field values from a single text chunk."""
    config = FieldDiscoveryConfig()
    client = _get_azure_client()
    deployment = _get_deployment(config)

    try:
        extractions = extract_fields_from_chunk(
            client, deployment, request.cluster_label, request.text, request.fields, config,
        )
    except Exception as e:
        raise HTTPException(500, f"Extraction failed: {e}")

    # Add grounding analysis
    result_fields = {}
    for fname, (value, confidence) in extractions.items():
        grounding = _grounding_detail(value, request.text) if value else {"level": "missing", "score": 0.0}
        result_fields[fname] = {
            "value": value,
            "confidence": confidence,
            "grounding": grounding,
        }

    return {
        "cluster_label": request.cluster_label,
        "fields": result_fields,
        "n_extracted": sum(1 for v in result_fields.values() if v["value"]),
        "n_fields": len(request.fields),
    }


@app.post("/api/v1/discover-extract")
async def discover_and_extract(request: DiscoverExtractRequest):
    """Full pipeline: discover fields for a cluster, then extract from chunks."""
    conn = _get_db(request.db_path)
    config = FieldDiscoveryConfig()
    client = _get_azure_client()
    deployment = _get_deployment(config)

    # Fetch cluster
    cluster = conn.execute(
        "SELECT cluster_id, label, chunk_count FROM clusters WHERE cluster_id = ?",
        [request.cluster_id],
    ).fetchone()
    if not cluster:
        conn.close()
        raise HTTPException(404, f"Cluster not found: {request.cluster_id}")

    cluster_id, cluster_label, chunk_count = cluster

    # Fetch chunks for discovery (broader sample)
    discovery_chunks = conn.execute("""
        SELECT ch.chunk_text FROM chunks ch
        JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
        WHERE ca.cluster_id = ?
        ORDER BY RANDOM() LIMIT 30
    """, [cluster_id]).fetchdf()["chunk_text"].tolist()

    # Fetch chunks for extraction (smaller, focused)
    if request.chunk_ids:
        placeholders = ",".join(["?"] * len(request.chunk_ids))
        extract_df = conn.execute(f"""
            SELECT ch.chunk_id, ch.chunk_text FROM chunks ch
            WHERE ch.chunk_id IN ({placeholders})
        """, request.chunk_ids).fetchdf()
    else:
        extract_df = conn.execute("""
            SELECT ch.chunk_id, ch.chunk_text FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ca.cluster_id = ?
            ORDER BY RANDOM() LIMIT ?
        """, [cluster_id, request.n_chunks]).fetchdf()

    conn.close()

    extract_chunk_ids = extract_df["chunk_id"].tolist()
    extract_texts = extract_df["chunk_text"].tolist()

    # Step 1: Discover
    t0 = time.time()
    method = request.method

    if method == DiscoveryMethod.ensemble:
        disc_result = discover_ensemble(cluster_label, discovery_chunks, config, client, deployment)
        fields = disc_result["fields"]
    elif method == DiscoveryMethod.standard_gpt:
        fields = discover_fields_for_cluster(client, deployment, cluster_label, discovery_chunks, config)
        disc_result = {"fields": fields, "methods_used": ["standard_gpt"]}
    elif method == DiscoveryMethod.rlm_v2:
        v2 = discover_fields_for_cluster_rlm_v2(cluster_label, discovery_chunks, config)
        fields = v2["fields"]
        disc_result = {"fields": fields, "methods_used": ["rlm_v2"], "v2_bonus": v2}
    else:
        fields = discover_fields_for_cluster(client, deployment, cluster_label, discovery_chunks, config)
        disc_result = {"fields": fields, "methods_used": [method.value]}

    disc_time = round(time.time() - t0, 2)

    # Step 2: Extract
    t0 = time.time()
    chunk_results = []
    for chunk_id, chunk_text in zip(extract_chunk_ids, extract_texts):
        try:
            ext = extract_fields_from_chunk(
                client, deployment, cluster_label, chunk_text, fields, config,
            )
        except Exception:
            ext = {}

        field_results = {}
        for f in fields:
            fname = f["name"]
            if fname in ext:
                value, confidence = ext[fname]
                grounding = _grounding_detail(value, chunk_text) if value else {"level": "missing", "score": 0.0}
                field_results[fname] = {"value": value, "confidence": confidence, "grounding": grounding}
            else:
                field_results[fname] = {"value": None, "confidence": 0.0, "grounding": {"level": "missing", "score": 0.0}}

        chunk_results.append({
            "chunk_id": chunk_id,
            "chunk_preview": chunk_text[:200],
            "fields": field_results,
        })

    extract_time = round(time.time() - t0, 2)

    # Step 3: Verify (optional)
    judge_results = None
    if request.verify and chunk_results:
        judge_results = await _run_judge_on_results(
            client, deployment, cluster_label, extract_texts, chunk_results,
        )

    # Aggregate metrics
    agg = _aggregate_extractions(chunk_results)

    return {
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "method": method.value,
        "discovery": {
            "n_fields": len(fields),
            "fields": fields,
            "time_s": disc_time,
            **({"methods_used": disc_result.get("methods_used")} if "methods_used" in disc_result else {}),
        },
        "extraction": {
            "n_chunks": len(chunk_results),
            "time_s": extract_time,
            "aggregate": agg,
            "chunks": chunk_results,
        },
        "judge": judge_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — JUDGE & CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/judge")
async def judge_extractions(request: JudgeRequest):
    """Judge extraction quality using LLM-as-a-judge."""
    from scripts.llm_judge_eval import judge_batch, compute_precision_recall

    client = _get_azure_client()
    deployment = _get_deployment(FieldDiscoveryConfig())

    verdicts = judge_batch(
        client, deployment,
        request.cluster_label,
        request.chunk_text,
        request.extractions,
    )

    metrics = compute_precision_recall(verdicts)

    return {
        "verdicts": verdicts,
        "metrics": metrics,
    }


@app.post("/api/v1/correct")
async def submit_correction(request: CorrectionRequest):
    """
    Submit a human correction. Builds the gold set for precision/recall tracking.
    Stores in the corrections table + updates extraction if applicable.
    """
    conn = _get_db(request.db_path, read_only=False)

    correction_id = f"cor_{uuid.uuid4().hex[:12]}"

    conn.execute("""
        INSERT INTO corrections (correction_id, cluster_id, chunk_id, field_name,
                                 original_value, corrected_value, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [
        correction_id,
        request.cluster_id,
        request.chunk_id,
        request.field_name,
        request.original_value,
        request.corrected_value,
        datetime.now(timezone.utc).isoformat(),
    ])

    conn.close()

    return {
        "correction_id": correction_id,
        "status": "saved",
        "message": "Correction stored. Will be used as gold label for eval.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — QUERY
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/clusters")
async def list_clusters(
    db_path: str | None = None,
    min_chunks: int = Query(default=5, description="Minimum chunk count"),
    limit: int = Query(default=50),
):
    """List clusters with metadata."""
    conn = _get_db(db_path)
    rows = conn.execute("""
        SELECT c.cluster_id, c.label, c.chunk_count, c.quality_score,
               COUNT(DISTINCT fd.field_id) as n_fields
        FROM clusters c
        LEFT JOIN field_definitions fd ON c.cluster_id = fd.cluster_id
        WHERE c.chunk_count >= ?
        GROUP BY c.cluster_id, c.label, c.chunk_count, c.quality_score
        ORDER BY c.chunk_count DESC
        LIMIT ?
    """, [min_chunks, limit]).fetchdf()
    conn.close()
    return {"clusters": rows.to_dict("records")}


@app.get("/api/v1/clusters/{cluster_id}")
async def get_cluster(cluster_id: str, db_path: str | None = None):
    """Get cluster detail with fields and sample chunks."""
    conn = _get_db(db_path)

    cluster = conn.execute(
        "SELECT * FROM clusters WHERE cluster_id = ?", [cluster_id]
    ).fetchdf()
    if cluster.empty:
        conn.close()
        raise HTTPException(404, f"Cluster not found: {cluster_id}")

    fields = conn.execute(
        "SELECT field_id, name, field_type, description FROM field_definitions WHERE cluster_id = ?",
        [cluster_id],
    ).fetchdf()

    sample_chunks = conn.execute("""
        SELECT ch.chunk_id, ch.chunk_text
        FROM chunks ch
        JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
        WHERE ca.cluster_id = ?
        ORDER BY RANDOM() LIMIT 5
    """, [cluster_id]).fetchdf()

    conn.close()

    return {
        "cluster": cluster.iloc[0].to_dict(),
        "fields": fields.to_dict("records"),
        "sample_chunks": sample_chunks.to_dict("records"),
    }


@app.get("/api/v1/clusters/{cluster_id}/fields")
async def get_cluster_fields(cluster_id: str, db_path: str | None = None):
    """Get field definitions for a cluster."""
    conn = _get_db(db_path)
    fields = conn.execute(
        "SELECT * FROM field_definitions WHERE cluster_id = ?",
        [cluster_id],
    ).fetchdf()
    conn.close()
    return {"fields": fields.to_dict("records")}


@app.get("/api/v1/agreements/{agreement_id}")
async def get_agreement(agreement_id: str, db_path: str | None = None):
    """Get agreement with its extractions."""
    conn = _get_db(db_path)

    agreement = conn.execute(
        "SELECT * FROM agreements WHERE agreement_id = ?", [agreement_id]
    ).fetchdf()
    if agreement.empty:
        conn.close()
        raise HTTPException(404, f"Agreement not found: {agreement_id}")

    extractions = conn.execute("""
        SELECT e.extraction_id, fd.name as field_name, e.value, e.confidence, e.was_corrected
        FROM extractions e
        JOIN field_definitions fd ON e.field_id = fd.field_id
        WHERE e.agreement_id = ?
        ORDER BY fd.name
    """, [agreement_id]).fetchdf()

    conn.close()

    return {
        "agreement": agreement.iloc[0].to_dict(),
        "extractions": extractions.to_dict("records"),
    }


@app.get("/api/v1/search")
async def semantic_search(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(default=10),
    db_path: str | None = None,
):
    """Semantic search across chunks using embedding similarity."""
    embedder = _get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]

    conn = _get_db(db_path)

    # Get chunks with embeddings
    chunks = conn.execute("""
        SELECT ch.chunk_id, ch.chunk_text, ch.embedding,
               ca.cluster_id, c.label as cluster_label
        FROM chunks ch
        JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
        JOIN clusters c ON ca.cluster_id = c.cluster_id
        LIMIT 5000
    """).fetchdf()
    conn.close()

    if chunks.empty:
        return {"results": []}

    # Compute similarities
    results = []
    for _, row in chunks.iterrows():
        emb_bytes = row["embedding"]
        if emb_bytes is None:
            continue
        try:
            chunk_emb = np.frombuffer(emb_bytes, dtype=np.float32)
            chunk_emb = chunk_emb / (np.linalg.norm(chunk_emb) + 1e-10)
            sim = float(np.dot(q_emb[:len(chunk_emb)], chunk_emb))
        except Exception:
            continue

        results.append({
            "chunk_id": row["chunk_id"],
            "chunk_text": row["chunk_text"][:500],
            "cluster_id": row["cluster_id"],
            "cluster_label": row["cluster_label"],
            "similarity": round(sim, 4),
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {"results": results[:top_k]}


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — EVAL
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/eval/metrics")
async def eval_metrics(db_path: str | None = None, gold_path: str | None = None):
    """Run the evaluation suite and return metrics."""
    from eval.runner import run_evaluation

    db = db_path or str(DB_PATH)
    gold = gold_path or str(Path(__file__).parent / "eval" / "gold_standard.json")
    if not Path(gold).exists():
        gold = None

    metrics = run_evaluation(db_path=db, gold_path=gold)
    return {"metrics": metrics}


@app.get("/api/v1/eval/compare")
async def eval_compare(db_path: str | None = None):
    """Return the latest method comparison results."""
    db_dir = Path(db_path).parent if db_path else DATA_DIR

    # Try v2 comparison first, fall back to v1
    for fname in ["extraction_value_comparison_v2.json", "extraction_value_comparison.json"]:
        path = db_dir / fname
        if path.exists():
            with open(path) as f:
                return json.load(f)

    # Try judge results
    judge_path = db_dir / "judge_eval_results.json"
    if judge_path.exists():
        with open(judge_path) as f:
            return json.load(f)

    raise HTTPException(404, "No comparison results found. Run a comparison first.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — PIPELINE & HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/pipeline/run")
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Trigger the full clustering pipeline in the background."""
    pdf_path = Path(request.pdf_dir)
    if not pdf_path.exists() or not pdf_path.is_dir():
        raise HTTPException(400, f"Invalid directory: {request.pdf_dir}")

    pdf_count = sum(1 for _ in pdf_path.rglob("*.pdf"))
    if pdf_count == 0:
        raise HTTPException(400, f"No PDFs found in {request.pdf_dir}")

    job_id = f"job_{uuid.uuid4().hex[:12]}"

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "account_id": request.account_id,
            "status": "queued",
            "progress": None,
            "result": None,
            "error": None,
        }

    background_tasks.add_task(_run_pipeline_bg, job_id, request)

    return {"job_id": job_id, "status": "queued", "pdf_count": pdf_count}


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


@app.get("/api/v1/health")
async def health():
    """Health check."""
    import torch

    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "device": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
        }

    # Check Azure OpenAI connectivity
    llm_ok = False
    try:
        client = _get_azure_client()
        deployment = _get_deployment(FieldDiscoveryConfig())
        llm_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "gpu": gpu_info,
        "llm_connected": llm_ok,
        "db_path": str(DB_PATH),
        "db_exists": DB_PATH.exists(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _grounding_detail(value: str, chunk_text: str) -> dict:
    if not value or not chunk_text:
        return {"level": "missing", "score": 0.0}
    v_lower = value.lower().strip()
    c_lower = chunk_text.lower()
    if v_lower in c_lower:
        return {"level": "verbatim", "score": 1.0}
    v_tokens = set(re.findall(r'\b\w+\b', v_lower))
    c_tokens = set(re.findall(r'\b\w+\b', c_lower))
    if not v_tokens:
        return {"level": "empty", "score": 0.0}
    overlap = v_tokens & c_tokens
    ratio = len(overlap) / len(v_tokens)
    if ratio >= 0.8:
        return {"level": "high_overlap", "score": 0.8}
    elif ratio >= 0.5:
        return {"level": "partial_overlap", "score": 0.5}
    elif ratio >= 0.2:
        return {"level": "weak_overlap", "score": 0.2}
    return {"level": "hallucinated", "score": 0.0}


def _aggregate_extractions(chunk_results: list[dict]) -> dict:
    """Compute aggregate fill rate, grounding stats from chunk results."""
    n_total = 0
    n_filled = 0
    grounding_counts = {}
    confidences = []

    for cr in chunk_results:
        for fname, fdata in cr["fields"].items():
            n_total += 1
            if fdata["value"] is not None:
                n_filled += 1
                level = fdata["grounding"]["level"]
                grounding_counts[level] = grounding_counts.get(level, 0) + 1
                if fdata.get("confidence"):
                    confidences.append(fdata["confidence"])

    return {
        "fill_rate": round(n_filled / n_total, 4) if n_total else 0,
        "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0,
        "n_total_slots": n_total,
        "n_filled": n_filled,
        "grounding_distribution": grounding_counts,
    }


async def _run_judge_on_results(client, deployment, cluster_label, chunk_texts, chunk_results):
    """Run LLM judge on extraction results and return precision/recall."""
    from scripts.llm_judge_eval import judge_batch, compute_precision_recall

    all_verdicts = []
    for cr, chunk_text in zip(chunk_results, chunk_texts):
        fields_to_judge = []
        for fname, fdata in cr["fields"].items():
            if fdata["value"] is not None:
                fields_to_judge.append({
                    "field_name": fname,
                    "extracted_value": str(fdata["value"]),
                })

        if fields_to_judge:
            verdicts = judge_batch(client, deployment, cluster_label, chunk_text, fields_to_judge)
            for v in verdicts:
                v["extracted_value"] = next(
                    (f["extracted_value"] for f in fields_to_judge if f["field_name"] == v.get("field_name")), None
                )
            all_verdicts.extend(verdicts)

    metrics = compute_precision_recall(all_verdicts)

    return {
        "metrics": metrics,
        "verdicts": all_verdicts,
    }


def _run_pipeline_bg(job_id: str, request: PipelineRequest):
    """Background pipeline runner."""
    from core.cluster_pipeline import run_cluster_pipeline

    with _jobs_lock:
        _jobs[job_id]["status"] = "running"

    try:
        result = run_cluster_pipeline(
            pdf_dir=request.pdf_dir,
            account_id=request.account_id,
            max_docs=request.max_docs,
            device=request.device,
        )

        serializable = {k: v for k, v in result.items() if k not in ("store", "lance_store")}

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = serializable

    except Exception as e:
        logger.exception(f"Pipeline job {job_id} failed")
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
