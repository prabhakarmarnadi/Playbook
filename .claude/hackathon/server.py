"""
FastAPI server for the contract extraction pipeline.

Endpoints:
  ── Ingestion ──
  POST /ingest/single         – Upload & process a single PDF (full pipeline)
  POST /ingest/bulk           – Upload & process multiple PDFs (full pipeline)

  ── Assignment (fast path) ──
  POST /assign/single         – Classify a single PDF into existing clusters
  POST /assign/bulk           – Classify multiple PDFs into existing clusters

  ── Core queries ──
  GET  /domains               – List domains
  GET  /agreements            – List agreements (filter: domain_id)
  GET  /clusters              – List clusters (filter: domain_id)
  GET  /clusters/{id}         – Single cluster detail
  GET  /fields                – List field definitions (filter: cluster_id)
  GET  /domains/{id}/fields   – List domain-level field definitions
  GET  /extractions           – Query extractions (filters: agreement_id, field_id, cluster_id)
  GET  /composites            – List composite/derived field definitions (filter: cluster_id)

  ── Chunks & Clauses ──
  GET  /chunks                – Get chunks (filter: agreement_id, cluster_id)
  GET  /clauses               – Get clauses (filter: agreement_id)

  ── Intents ──
  GET  /intents/types         – List intent type taxonomy
  GET  /intents               – Query clause-level intents (filters: agreement_id, clause_id, intent_type_id)

  ── Corrections ──
  GET  /corrections           – Correction history (filter: field_id)

  ── Pipeline stats ──
  GET  /stats                 – Aggregate pipeline statistics
  GET  /novel-documents       – Buffered novel/unassigned documents

  ── Feedback & Personalization ──
  POST /feedback/events       – Record a feedback event
  GET  /feedback/events       – Query feedback history
  GET  /feedback/summary      – Tenant feedback summary
  GET  /tenants/{id}/config   – Tenant threshold config
  PATCH /tenants/{id}/config  – Update tenant config
  GET  /virtual-clusters      – List tenant virtual clusters
  GET  /intent-overrides      – List tenant intent label overrides
  GET  /personalization/context – Full personalization context for inference

  ── Ops ──
  GET  /health                – Liveness check

Usage:
  cd /home/azureuser/clustering-v2/hackathon
  source /home/azureuser/.venv/bin/activate
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Contract Extraction Pipeline API",
    version="0.2.0",
    description="Ingest contract PDFs and extract structured fields via clustering + RAG.",
)
logger = logging.getLogger("server")


# ── Request models ─────────────────────────────────────────────────────────────

class FeedbackEventRequest(BaseModel):
    user_id: str
    entity_type: str
    entity_id: str
    action: str  # correct | reject | refine | approve | link | unlink
    old_value: dict | None = None
    new_value: dict | None = None
    context: dict | None = None
    source_doc_id: str | None = None
    session_id: str | None = None


class TenantConfigUpdate(BaseModel):
    high_confidence_threshold: float | None = None
    tentative_threshold: float | None = None
    outlier_min_similarity: float | None = None
    intent_match_threshold: float | None = None
    intent_temperature: float | None = None
    edge_decay_half_life_days: int | None = None
    auto_apply_corrections: bool | None = None

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_ACCOUNT = "public_contracts_perf"
DEFAULT_DB = "data/public_contracts_perf.duckdb"
DEFAULT_LANCE = "data/public_contracts_perf.lance"


class _NumpySafeEncoder(json.JSONEncoder):
    """Handle numpy/pandas/DuckDB types that come out of .fetchdf().to_dict()."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (bytes, bytearray)):
            return None  # strip binary blobs (embeddings)
        if hasattr(o, 'isoformat'):
            return o.isoformat()
        return super().default(o)


def _json_response(data):
    """Serialize store results safely (handles numpy types)."""
    return JSONResponse(content=json.loads(json.dumps(data, cls=_NumpySafeEncoder)))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_uploads_to_tmpdir(files: list[UploadFile]) -> Path:
    """Write uploaded PDFs to a temp directory and return the path."""
    tmp = Path(tempfile.mkdtemp(prefix="pipeline_upload_"))
    for f in files:
        if not f.filename:
            continue
        dest = tmp / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
    return tmp


def _build_config(account_id: str | None = None):
    """Build a MinimalPipelineConfig pointing at the right DB."""
    from core.minimal_pipeline import MinimalPipelineConfig

    acct = account_id or DEFAULT_ACCOUNT
    return MinimalPipelineConfig(
        db_path=f"data/{acct}.duckdb",
        lance_path=f"data/{acct}.lance",
        use_late_interaction=True,
        widen_retrieval_scope=True,
        enable_confidence_gate=True,
        confidence_gate_threshold=0.55,
        batch_small_clusters=True,
        min_cluster_chunks=10,
    ), acct


def _get_store(account_id: str | None = None):
    import duckdb
    from core.store import ClusteringStore
    acct = account_id or DEFAULT_ACCOUNT
    db_path = Path(__file__).parent / "data" / f"{acct}.duckdb"
    if not db_path.exists():
        raise HTTPException(404, f"Database not found for account '{acct}'")
    try:
        store = object.__new__(ClusteringStore)
        store.db_path = str(db_path)
        store.conn = duckdb.connect(str(db_path), read_only=True)
        return store
    except duckdb.IOException:
        raise HTTPException(
            503,
            f"Database '{acct}' is locked by a running pipeline. "
            "Try again after the run completes, or use a different account_id.",
        )


def _get_feedback_store(tenant_id: str):
    """Open the feedback sidecar DB for a tenant."""
    from core.feedback_store import FeedbackStore
    return FeedbackStore(tenant_id=tenant_id, data_dir=str(Path(__file__).parent / "data"))


def _get_personalization_engine(tenant_id: str, account_id: str | None = None):
    """Build a PersonalizationEngine wired to both feedback and main DB."""
    from core.personalization import PersonalizationEngine
    fb = _get_feedback_store(tenant_id)
    acct = account_id or DEFAULT_ACCOUNT
    main_db = str(Path(__file__).parent / "data" / f"{acct}.duckdb")
    return PersonalizationEngine(fb, main_db_path=main_db)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Ingest: full pipeline (parse → cluster → discover → extract) ──────────────

@app.post("/ingest/single")
async def ingest_single(
    file: UploadFile = File(...),
    account_id: str | None = Query(None, description="Account/tenant ID"),
):
    """Upload a single PDF and run the full extraction pipeline on it."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    tmp = _save_uploads_to_tmpdir([file])
    try:
        from core.minimal_pipeline import run_minimal_pipeline
        config, acct = _build_config(account_id)
        result = await run_minimal_pipeline(
            pdf_dir=str(tmp),
            account_id=acct,
            config=config,
        )
        return JSONResponse(content=result)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/ingest/bulk")
async def ingest_bulk(
    files: list[UploadFile] = File(...),
    account_id: str | None = Query(None, description="Account/tenant ID"),
):
    """Upload multiple PDFs and run the full extraction pipeline on them."""
    pdf_files = [f for f in files if f.filename and f.filename.lower().endswith(".pdf")]
    if not pdf_files:
        raise HTTPException(400, "No PDF files provided")

    tmp = _save_uploads_to_tmpdir(pdf_files)
    try:
        from core.minimal_pipeline import run_minimal_pipeline
        config, acct = _build_config(account_id)
        result = await run_minimal_pipeline(
            pdf_dir=str(tmp),
            account_id=acct,
            config=config,
        )
        return JSONResponse(content=result)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Assign: incremental classification into existing clusters ──────────────────

@app.post("/assign/single")
async def assign_single(
    file: UploadFile = File(...),
    account_id: str | None = Query(None, description="Account/tenant ID"),
):
    """Classify a single PDF against existing clusters (fast path, no re-extraction)."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    tmp = _save_uploads_to_tmpdir([file])
    try:
        from core.minimal_pipeline import assign_new_documents
        config, acct = _build_config(account_id)
        pdf_path = next(tmp.glob("*.pdf"))
        results = await assign_new_documents(
            pdf_paths=[str(pdf_path)],
            account_id=acct,
            config=config,
        )
        return JSONResponse(content=results)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/assign/bulk")
async def assign_bulk(
    files: list[UploadFile] = File(...),
    account_id: str | None = Query(None, description="Account/tenant ID"),
):
    """Classify multiple PDFs against existing clusters (fast path)."""
    pdf_files = [f for f in files if f.filename and f.filename.lower().endswith(".pdf")]
    if not pdf_files:
        raise HTTPException(400, "No PDF files provided")

    tmp = _save_uploads_to_tmpdir(pdf_files)
    try:
        from core.minimal_pipeline import assign_new_documents
        config, acct = _build_config(account_id)
        pdf_paths = [str(p) for p in tmp.glob("*.pdf")]
        results = await assign_new_documents(
            pdf_paths=pdf_paths,
            account_id=acct,
            config=config,
        )
        return JSONResponse(content=results)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Query endpoints ────────────────────────────────────────────────────────────

@app.get("/extractions")
async def get_extractions(
    agreement_id: str | None = Query(None),
    field_id: str | None = Query(None),
    cluster_id: str | None = Query(None),
    account_id: str | None = Query(None),
):
    """Retrieve extractions with optional filters."""
    store = _get_store(account_id)
    try:
        rows = store.get_extractions(
            agreement_id=agreement_id,
            field_id=field_id,
            cluster_id=cluster_id,
        )
        return _json_response(rows)
    finally:
        store.close()


@app.get("/clusters")
async def get_clusters(
    domain_id: str | None = Query(None),
    account_id: str | None = Query(None),
):
    """List clusters, optionally filtered by domain."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_clusters(domain_id=domain_id))
    finally:
        store.close()


@app.get("/fields")
async def get_fields(
    cluster_id: str | None = Query(None),
    account_id: str | None = Query(None),
):
    """List field definitions, optionally filtered by cluster."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_fields(cluster_id=cluster_id))
    finally:
        store.close()


@app.get("/domains")
async def get_domains(account_id: str | None = Query(None)):
    """List all discovered domains."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_domains())
    finally:
        store.close()


@app.get("/agreements")
async def get_agreements(
    domain_id: str | None = Query(None),
    account_id: str | None = Query(None),
):
    """List agreements, optionally filtered by domain."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_agreements(domain_id=domain_id))
    finally:
        store.close()


# ── Single cluster detail ──────────────────────────────────────────────────────

@app.get("/clusters/{cluster_id}")
async def get_cluster_detail(
    cluster_id: str,
    account_id: str | None = Query(None),
):
    """Get full detail for a single cluster (label, keywords, quality score, etc.)."""
    store = _get_store(account_id)
    try:
        row = store.get_cluster(cluster_id)
        if not row:
            raise HTTPException(404, f"Cluster '{cluster_id}' not found")
        return _json_response(row)
    finally:
        store.close()


# ── Domain-level field definitions (two-tier architecture) ─────────────────────

@app.get("/domains/{domain_id}/fields")
async def get_domain_fields(
    domain_id: str,
    account_id: str | None = Query(None),
):
    """List domain-level universal field definitions (seeded across clusters)."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_domain_fields(domain_id))
    except Exception as e:
        if "does not exist" in str(e):
            return _json_response([])
        raise
    finally:
        store.close()


# ── Composite / derived field definitions ──────────────────────────────────────

@app.get("/composites")
async def get_composites(
    cluster_id: str | None = Query(None),
    account_id: str | None = Query(None),
):
    """List composite (derived) field definitions."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_composites(cluster_id=cluster_id))
    finally:
        store.close()


# ── Chunks & Clauses ──────────────────────────────────────────────────────────

@app.get("/chunks")
async def get_chunks(
    agreement_id: str | None = Query(None),
    cluster_id: str | None = Query(None),
    account_id: str | None = Query(None),
):
    """Get document chunks, filtered by agreement or cluster."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_chunks(
            agreement_id=agreement_id,
            cluster_id=cluster_id,
        ))
    finally:
        store.close()


@app.get("/clauses")
async def get_clauses(
    agreement_id: str | None = Query(None),
    account_id: str | None = Query(None),
):
    """Get parsed clauses for an agreement."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_clauses(agreement_id=agreement_id))
    finally:
        store.close()


# ── Intent taxonomy & clause intents ──────────────────────────────────────────

@app.get("/intents/types")
async def get_intent_types(account_id: str | None = Query(None)):
    """List the intent type taxonomy (labels, aliases, occurrence counts)."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_intent_types())
    finally:
        store.close()


@app.get("/intents")
async def get_clause_intents(
    agreement_id: str | None = Query(None),
    clause_id: str | None = Query(None),
    clause_type_id: str | None = Query(None),
    intent_type_id: str | None = Query(None),
    account_id: str | None = Query(None),
):
    """Query clause-level intents with optional filters."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_clause_intents(
            clause_type_id=clause_type_id,
            agreement_id=agreement_id,
            clause_id=clause_id,
            intent_type_id=intent_type_id,
        ))
    finally:
        store.close()


# ── Corrections history ───────────────────────────────────────────────────────

@app.get("/corrections")
async def get_corrections(
    field_id: str | None = Query(None),
    account_id: str | None = Query(None),
):
    """Get extraction correction history, optionally filtered by field."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_corrections(field_id=field_id))
    finally:
        store.close()


# ── Pipeline statistics ───────────────────────────────────────────────────────

@app.get("/stats")
async def get_stats(account_id: str | None = Query(None)):
    """Aggregate pipeline statistics (counts, coverage)."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_stats())
    finally:
        store.close()


# ── Novel document buffer ─────────────────────────────────────────────────────

@app.get("/novel-documents")
async def get_novel_documents(account_id: str | None = Query(None)):
    """List documents buffered as 'novel' (not matching any cluster)."""
    store = _get_store(account_id)
    try:
        return _json_response(store.get_novel_documents())
    finally:
        store.close()


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK & PERSONALIZATION
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/feedback/events")
async def record_feedback(
    body: FeedbackEventRequest,
    tenant_id: str = Query(..., description="Tenant ID for feedback isolation"),
):
    """Record a feedback event (correct, reject, approve, etc.) and update weights."""
    fb = _get_feedback_store(tenant_id)
    try:
        event_id = fb.record_feedback(
            user_id=body.user_id,
            entity_type=body.entity_type,
            entity_id=body.entity_id,
            action=body.action,
            old_value=body.old_value,
            new_value=body.new_value,
            context=body.context,
            source_doc_id=body.source_doc_id,
            session_id=body.session_id,
        )
        return {"event_id": event_id}
    finally:
        fb.close()


@app.get("/feedback/events")
async def get_feedback_history(
    tenant_id: str = Query(...),
    entity_type: str | None = Query(None),
    entity_id: str | None = Query(None),
    limit: int = Query(100, le=1000),
):
    """Query feedback event history for a tenant."""
    fb = _get_feedback_store(tenant_id)
    try:
        return _json_response(fb.get_feedback_history(
            entity_type=entity_type,
            entity_id=entity_id,
            limit=limit,
        ))
    finally:
        fb.close()


@app.get("/feedback/summary")
async def get_feedback_summary(tenant_id: str = Query(...)):
    """Summary counts: events, weights, virtual clusters, overrides."""
    fb = _get_feedback_store(tenant_id)
    try:
        return _json_response(fb.summary())
    finally:
        fb.close()


# ── Tenant configuration ──────────────────────────────────────────────────────

@app.get("/tenants/{tenant_id}/config")
async def get_tenant_config(tenant_id: str):
    """Get tenant-specific threshold configuration."""
    fb = _get_feedback_store(tenant_id)
    try:
        return _json_response(fb.get_tenant_config())
    finally:
        fb.close()


@app.patch("/tenants/{tenant_id}/config")
async def update_tenant_config(tenant_id: str, body: TenantConfigUpdate):
    """Update tenant thresholds (confidence, outlier, intent settings)."""
    fb = _get_feedback_store(tenant_id)
    try:
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        if not updates:
            raise HTTPException(400, "No fields to update")
        fb.update_tenant_config(**updates)
        return _json_response(fb.get_tenant_config())
    finally:
        fb.close()


# ── Virtual clusters (tenant-specific outlier management) ──────────────────────

@app.get("/virtual-clusters")
async def get_virtual_clusters(tenant_id: str = Query(...)):
    """List tenant-specific virtual clusters."""
    fb = _get_feedback_store(tenant_id)
    try:
        return _json_response(fb.get_virtual_clusters())
    finally:
        fb.close()


# ── Intent overrides (tenant vocabulary) ───────────────────────────────────────

@app.get("/intent-overrides")
async def get_intent_overrides(tenant_id: str = Query(...)):
    """List tenant intent label overrides (global→tenant label mapping)."""
    fb = _get_feedback_store(tenant_id)
    try:
        return _json_response(fb.get_intent_overrides())
    finally:
        fb.close()


# ── Personalization context (full inference-time bundle) ───────────────────────

@app.get("/personalization/context")
async def get_personalization_context(
    tenant_id: str = Query(...),
    account_id: str | None = Query(None),
    clause_type_id: str | None = Query(None),
    agreement_id: str | None = Query(None),
):
    """Get the full personalization context used during inference."""
    from dataclasses import asdict
    pe = _get_personalization_engine(tenant_id, account_id)
    try:
        ctx = pe.get_context(
            clause_type_id=clause_type_id,
            agreement_id=agreement_id,
        )
        result = asdict(ctx)
        # Convert set to list for JSON
        result["approved_entities"] = list(result.get("approved_entities", []))
        # Convert tuple keys in edge_weights to string keys
        result["edge_weights"] = {
            f"{k[0]}→{k[1]}": v
            for k, v in result.get("edge_weights", {}).items()
        }
        return _json_response(result)
    finally:
        pe.store.close()
