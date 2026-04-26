"""
FastAPI server for the clustering pipeline.

Endpoints:
    POST /cluster            Trigger featurizer + clustering on a PDF directory
    GET  /runs/{account_id}  List pipeline runs for an account
    GET  /runs/{account_id}/{run_id}  Get run manifest + stats
    GET  /health             Health check

Usage:
    poetry run uvicorn api_server:app --host 0.0.0.0 --port 8000
"""
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from config import DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

app = FastAPI(
    title="Clustering V2 API",
    description="Featurizer + clustering pipeline for contract agreements",
    version="0.1.0",
)

# ── In-memory job tracker ──────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

ARTIFACTS_DIR = DATA_DIR / "artifacts"


# ── Request / Response models ─────────────────────────────────────────────────

class ClusterRequest(BaseModel):
    pdf_dir: str = Field(..., description="Absolute path to directory containing PDF files")
    account_id: str = Field(default="default", description="Account/tenant ID for isolation")
    max_docs: int | None = Field(default=None, description="Limit number of documents to process")
    device: str = Field(default="cuda", description="Torch device: cuda or cpu")
    use_semantic_chunking: bool = Field(default=True, description="Use semantic chunking (slower but better). False = sliding window.")
    use_rlm: bool = Field(default=False, description="Use dspy.RLM for field discovery (slower but explores all chunks)")

class ClusterResponse(BaseModel):
    job_id: str
    account_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    account_id: str
    status: str  # queued, running, completed, failed
    progress: str | None = None
    result: dict | None = None
    error: str | None = None


# ── Background worker ─────────────────────────────────────────────────────────

def _run_pipeline_job(job_id: str, request: ClusterRequest):
    """Run the pipeline in background and update job status."""
    from core.cluster_pipeline import run_cluster_pipeline

    def update_progress(stage: str, msg: str):
        with _jobs_lock:
            _jobs[job_id]["progress"] = f"[{stage}] {msg}"

    with _jobs_lock:
        _jobs[job_id]["status"] = "running"

    try:
        result = run_cluster_pipeline(
            pdf_dir=request.pdf_dir,
            account_id=request.account_id,
            max_docs=request.max_docs,
            device=request.device,
            use_semantic_chunking=request.use_semantic_chunking,
            use_rlm=request.use_rlm,
            callback=update_progress,
        )

        # Remove non-serializable objects from result
        serializable = {
            k: v for k, v in result.items()
            if k not in ("store", "lance_store")
        }

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = serializable

    except Exception as e:
        logger.exception(f"Pipeline job {job_id} failed")
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/cluster", response_model=ClusterResponse)
async def trigger_clustering(request: ClusterRequest, background_tasks: BackgroundTasks):
    """
    Trigger the featurizer + clustering pipeline on a PDF directory.

    The pipeline runs in the background. Use GET /jobs/{job_id} to check status.
    """
    pdf_path = Path(request.pdf_dir)
    if not pdf_path.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.pdf_dir}")
    if not pdf_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {request.pdf_dir}")

    # Check for PDFs
    pdf_count = sum(1 for _ in pdf_path.glob("*.pdf"))
    if pdf_count == 0:
        # Try recursive
        pdf_count = sum(1 for _ in pdf_path.rglob("*.pdf"))
    if pdf_count == 0:
        raise HTTPException(status_code=400, detail=f"No PDF files found in {request.pdf_dir}")

    import uuid
    job_id = f"job_{uuid.uuid4().hex[:12]}"

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "account_id": request.account_id,
            "status": "queued",
            "progress": None,
            "result": None,
            "error": None,
            "pdf_dir": request.pdf_dir,
            "pdf_count": pdf_count,
        }

    background_tasks.add_task(_run_pipeline_job, job_id, request)

    return ClusterResponse(
        job_id=job_id,
        account_id=request.account_id,
        status="queued",
        message=f"Pipeline queued for {pdf_count} PDFs in {request.pdf_dir}",
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a pipeline job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return JobStatus(**{k: job[k] for k in JobStatus.model_fields if k in job})


@app.get("/jobs")
async def list_jobs(account_id: str | None = None):
    """List all pipeline jobs, optionally filtered by account_id."""
    with _jobs_lock:
        jobs = list(_jobs.values())
    if account_id:
        jobs = [j for j in jobs if j.get("account_id") == account_id]
    return {"jobs": jobs}


@app.get("/runs/{account_id}")
async def list_runs(account_id: str):
    """List all pipeline runs (from artifacts on disk) for an account."""
    account_dir = ARTIFACTS_DIR / account_id
    if not account_dir.exists():
        return {"account_id": account_id, "runs": []}

    runs = []
    for run_dir in sorted(account_dir.iterdir(), reverse=True):
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)
            runs.append({
                "run_id": manifest.get("run_id"),
                "n_documents": manifest.get("n_documents"),
                "n_domains": manifest.get("n_domains"),
                "n_clusters": manifest.get("n_clusters"),
                "elapsed_seconds": manifest.get("elapsed_seconds"),
                "artifacts_dir": str(run_dir),
            })

    return {"account_id": account_id, "runs": runs}


@app.get("/runs/{account_id}/{run_id}")
async def get_run(account_id: str, run_id: str):
    """Get the full manifest for a specific pipeline run."""
    manifest_path = ARTIFACTS_DIR / account_id / run_id / "run_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {account_id}/{run_id}")

    import json
    with open(manifest_path) as f:
        return json.load(f)


@app.get("/health")
async def health():
    """Health check with GPU and cuML status."""
    import torch
    from core.cuml_wrappers import cuml_available

    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuml_available": cuml_available(),
    }
