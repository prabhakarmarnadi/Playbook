"""
Scalable 3-Phase Pipeline — Feature Processing → Clustering → Extraction.
==========================================================================

Designed for 1M documents / 10K accounts. Self-contained architecture that
any agentic coding mechanism can implement from scratch.

Architecture Overview:
─────────────────────────────────────────────────────────────────────────

  PHASE 0 ─ FEATURE PROCESSING (embarrassingly parallel)
  ├── 0a. Ingest        : blob store → raw text (per-doc, stateless)
  ├── 0b. Chunk         : structural clause split + semantic overlap
  ├── 0c. Embed         : nomic-embed-text-v1.5 GPU batches (768-dim)
  ├── 0d. NL Index      : heading/section structure (per-doc, one-time)
  └── 0e. Persist       : pgvector + DuckDB (account-isolated)

  PHASE 1 ─ CLUSTERING (per-account, GPU-accelerated)
  ├── 1a. EVoC + Optuna : cosine-native clustering, 30-trial TPE
  ├── 1b. Layer Select  : silhouette + cosine scoring → best layer
  ├── 1c. Merge + Dedup : 0.96 cosine near-dup merge
  ├── 1d. Outlier Recov : iterative re-cluster or soft-assign
  ├── 1e. Label Synth   : LLM label synthesis per cluster
  └── 1f. Persist       : cluster assignments + centroids + metadata

  PHASE 2 ─ DISCOVERY + EXTRACTION (per-cluster, LLM-intensive)
  ├── 2a. RLM V3 Disc   : agentic REPL field discovery (+ inline extract)
  ├── 2b. Field Dedup    : cross-cluster cosine merge, universal fields
  ├── 2c. RAG Extract    : hybrid retrieval + agentic retry + reviewer
  ├── 2d. Normalize      : type-aware value normalization
  └── 2e. Persist        : field defs + extractions + feedback hooks

Scaling Strategy:
─────────────────────────────────────────────────────────────────────────

  Layer           │ Unit of Work        │ Parallelism      │ Bottleneck
  ────────────────┼─────────────────────┼──────────────────┼──────────
  Phase 0         │ Per document        │ 100s of workers  │ GPU embed
  Phase 1         │ Per account         │ 10s of accounts  │ Optuna trials
  Phase 2         │ Per cluster         │ 10s per account  │ LLM rate limit

  Documents → micro-batches (256 docs) → Phase 0 workers (stateless)
  Accounts  → Phase 1 job queue (one job per account, GPU-scheduled)
  Clusters  → Phase 2 semaphore pool (bounded LLM concurrency)

Checkpoint Contract:
─────────────────────────────────────────────────────────────────────────

  Every phase writes a checkpoint on completion. Restart skips completed
  phases. Each unit of work is idempotent: re-running produces the same
  output. Checkpoint = (phase, account_id, run_id, status, artifact_path).

Usage:
  # CLI
  python -m core.scalable_pipeline --accounts acct_001,acct_002 --phase all

  # Python
  from core.scalable_pipeline import ScalablePipeline, ScalableConfig
  pipeline = ScalablePipeline(config)
  result = await pipeline.run(account_ids=["acct_001"])

  # Single-phase
  result = await pipeline.run_phase0(account_id="acct_001", doc_ids=[...])
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════


class Phase(str, Enum):
    FEATURE_PROCESSING = "phase0"
    CLUSTERING = "phase1"
    EXTRACTION = "phase2"


class ExtractionBackend(str, Enum):
    RAG = "rag"
    SEQUENTIAL = "sequential"


class DiscoveryMode(str, Enum):
    STANDARD = "standard"
    RLM_V3 = "rlm_v3"
    RLM_V3_UNIFIED = "rlm_v3_unified"
    HYBRID = "hybrid"


class StorageBackend(str, Enum):
    DUCKDB = "duckdb"
    POSTGRES = "postgres"


@dataclass
class ScalableConfig:
    """
    Complete configuration for the 3-phase pipeline.

    Designed so an agentic system can read this config, understand every
    parameter, and implement each phase independently.
    """

    # ── Global ──
    device: str = "cuda"
    log_level: str = "INFO"
    data_dir: str = "data"
    artifacts_dir: str = "data/artifacts"
    blob_connection_string: str | None = None
    blob_container: str = "pipeline-v2"

    # ── Storage Backend ──
    storage_backend: str = "postgres"       # postgres | duckdb
    pg_dsn: str = field(default_factory=lambda: os.getenv(
        "PG_DSN", "postgresql://localhost:5432/clustering_v2"
    ))
    pg_min_pool: int = 5
    pg_max_pool: int = 50
    pg_hnsw_m: int = 16
    pg_hnsw_ef_construction: int = 128
    pg_hnsw_ef_search: int = 100
    pg_distance_metric: str = "cosine"     # cosine | l2 | inner_product
    pg_use_pg_search: bool = False          # ParadeDB BM25 (if installed)

    # ── Phase 0: Feature Processing ──
    # Ingest
    max_docs_per_batch: int = 256          # micro-batch size for parallel ingest
    pdf_parse_workers: int = 8             # concurrent PDF parsers

    # Chunking
    chunk_strategy: str = "semantic"       # semantic | sliding_window
    embedding_chunk_size: int = 2000       # tokens, for embedding/clustering
    embedding_chunk_overlap: int = 100
    extraction_chunk_size: int = 4000      # tokens, for sequential extraction
    clause_split: bool = True              # structural clause boundary detection
    max_chunks_per_doc: int = 80
    min_chunk_chars: int = 100
    max_chunk_chars: int = 2000
    cohesion_threshold: float = 0.75       # sub-split if cosine < this

    # Embedding
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dim: int = 768
    embedding_batch_size: int = 64         # GPU batch size
    embedding_prefix_index: str = "search_document"
    embedding_prefix_query: str = "search_query"

    # NL Index
    build_nl_index: bool = True

    # ── Phase 1: Clustering ──
    clustering_algorithm: str = "evoc"     # evoc | bertopic
    optuna_trials: int = 30
    evoc_min_cluster_size_range: tuple[int, int] = (3, 25)
    evoc_noise_level_range: tuple[float, float] = (0.1, 0.6)
    evoc_n_neighbors_range: tuple[int, int] = (5, 30)
    evoc_min_samples_range: tuple[int, int] = (1, 8)
    merge_threshold: float = 0.96          # near-dup cluster merge
    outlier_strategy: str = "iterative"    # iterative | soft | nearest
    outlier_min_similarity: float = 0.35
    max_outlier_ratio: float = 0.20
    min_clusters: int = 5
    label_synthesis_model: str = "gpt-5.4-nano"

    # ── Phase 2: Discovery + Extraction ──
    discovery_mode: str = "rlm_v3_unified"
    extraction_backend: str = "rag"

    # RLM V3 Discovery
    rlm_max_iterations: int = 25
    rlm_max_llm_calls: int = 50
    rlm_max_sample_chunks: int = 30
    max_fields_per_cluster: int = 15
    rlm_briefing_max_tokens: int = 800

    # Field Dedup
    field_dedup_threshold: float = 0.85    # cosine merge for cross-cluster dedup
    universal_field_min_clusters: int = 3  # appear in N+ clusters → universal

    # RAG Extraction
    cochran_z: float = 1.96
    cochran_e: float = 0.15
    max_sample_docs: int = 10
    rag_max_retry_rounds: int = 3
    rag_enable_reviewer: bool = True
    rag_semantic_top_k: int = 10
    rag_final_top_k: int = 3

    # Sequential Extraction (alternative backend)
    seq_carryforward: bool = True
    seq_judge_model: str = "gpt-5.4"
    seq_extractor_model: str = "gpt-5.4-nano"

    # Model Routing
    complex_model: str = "gpt-5.4"
    simple_model: str = "gpt-5.4-nano"
    min_extraction_confidence: float = 0.4

    # Concurrency
    max_llm_concurrency: int = 8           # global semaphore for LLM calls
    max_cluster_concurrency: int = 4       # parallel clusters in Phase 2
    max_doc_concurrency: int = 5           # parallel docs per cluster
    max_field_concurrency: int = 8         # parallel fields per doc

    # Checkpointing
    checkpoint_enabled: bool = True
    checkpoint_store: str = "duckdb"       # duckdb | blob | filesystem


# ══════════════════════════════════════════════════════════════════════════════
# §2  CHECKPOINT MANAGER
# ══════════════════════════════════════════════════════════════════════════════


class CheckpointStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Checkpoint:
    """Represents completion state of one unit of work."""
    phase: str
    account_id: str
    run_id: str
    unit_id: str                          # doc_id, cluster_id, etc.
    status: CheckpointStatus
    artifact_path: str | None = None
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    error: str | None = None


class CheckpointManager:
    """
    Manages idempotent phase execution via checkpoint tracking.

    Contract:
      - Before starting work: check if checkpoint exists → skip if COMPLETED
      - After completing work: write COMPLETED checkpoint
      - On failure: write FAILED checkpoint with error
      - Restart: re-runs only non-COMPLETED units

    Storage backends: DuckDB (default), filesystem, blob.
    """

    def __init__(self, store_path: str | Path, backend: str = "duckdb"):
        self._backend = backend
        self._store_path = Path(store_path)
        self._checkpoints: dict[str, Checkpoint] = {}  # key = f"{phase}:{account}:{unit}"
        self._conn = None

        if backend == "duckdb":
            self._init_duckdb()
        elif backend == "filesystem":
            self._store_path.mkdir(parents=True, exist_ok=True)

    def _init_duckdb(self):
        import duckdb
        self._conn = duckdb.connect(str(self._store_path / "checkpoints.duckdb"))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_key VARCHAR PRIMARY KEY,
                phase VARCHAR NOT NULL,
                account_id VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                unit_id VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                artifact_path VARCHAR,
                metadata JSON,
                timestamp DOUBLE,
                error VARCHAR
            )
        """)

    def _key(self, phase: str, account_id: str, unit_id: str) -> str:
        return f"{phase}:{account_id}:{unit_id}"

    def is_completed(self, phase: str, account_id: str, unit_id: str) -> bool:
        key = self._key(phase, account_id, unit_id)

        if self._backend == "duckdb" and self._conn:
            row = self._conn.execute(
                "SELECT status FROM checkpoints WHERE checkpoint_key = ?", [key]
            ).fetchone()
            return row is not None and row[0] == CheckpointStatus.COMPLETED.value

        return (
            key in self._checkpoints
            and self._checkpoints[key].status == CheckpointStatus.COMPLETED
        )

    def mark_started(self, phase: str, account_id: str, run_id: str, unit_id: str):
        self._write(Checkpoint(
            phase=phase, account_id=account_id, run_id=run_id,
            unit_id=unit_id, status=CheckpointStatus.IN_PROGRESS,
        ))

    def mark_completed(
        self,
        phase: str,
        account_id: str,
        run_id: str,
        unit_id: str,
        artifact_path: str | None = None,
        metadata: dict | None = None,
    ):
        self._write(Checkpoint(
            phase=phase, account_id=account_id, run_id=run_id,
            unit_id=unit_id, status=CheckpointStatus.COMPLETED,
            artifact_path=artifact_path, metadata=metadata or {},
        ))

    def mark_failed(
        self, phase: str, account_id: str, run_id: str, unit_id: str, error: str,
    ):
        self._write(Checkpoint(
            phase=phase, account_id=account_id, run_id=run_id,
            unit_id=unit_id, status=CheckpointStatus.FAILED, error=error,
        ))

    def get_incomplete(self, phase: str, account_id: str) -> list[str]:
        """Return unit_ids that are NOT completed for a phase + account."""
        if self._backend == "duckdb" and self._conn:
            rows = self._conn.execute(
                "SELECT unit_id FROM checkpoints WHERE phase = ? AND account_id = ? AND status != ?",
                [phase, account_id, CheckpointStatus.COMPLETED.value],
            ).fetchall()
            return [r[0] for r in rows]
        return [
            cp.unit_id for key, cp in self._checkpoints.items()
            if cp.phase == phase and cp.account_id == account_id
            and cp.status != CheckpointStatus.COMPLETED
        ]

    def phase_completed(self, phase: str, account_id: str) -> bool:
        """Check if an entire phase is marked complete for an account."""
        key = self._key(phase, account_id, "__phase__")
        return self.is_completed(phase, account_id, "__phase__")

    def mark_phase_completed(self, phase: str, account_id: str, run_id: str, metadata: dict | None = None):
        self.mark_completed(phase, account_id, run_id, "__phase__", metadata=metadata)

    def _write(self, cp: Checkpoint):
        key = self._key(cp.phase, cp.account_id, cp.unit_id)
        self._checkpoints[key] = cp

        if self._backend == "duckdb" and self._conn:
            self._conn.execute("""
                INSERT OR REPLACE INTO checkpoints
                (checkpoint_key, phase, account_id, run_id, unit_id, status,
                 artifact_path, metadata, timestamp, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                key, cp.phase, cp.account_id, cp.run_id, cp.unit_id,
                cp.status.value, cp.artifact_path,
                json.dumps(cp.metadata), cp.timestamp, cp.error,
            ])

    def close(self):
        if self._conn:
            self._conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# §3  STORE FACTORY — Account-isolated storage
# ══════════════════════════════════════════════════════════════════════════════


class StoreFactory:
    """
    Creates account-isolated store instances.

    Supports two backends:
      - postgres (default): PG schema-per-account via asyncpg + pgvector
      - duckdb: file-per-account DuckDB + LanceDB (dev/test)

    For Postgres:
      - Connection pool is shared across all accounts (no FD exhaustion)
      - pgvector HNSW indexes for ANN search (pre-filtered)
      - tsvector GIN indexes for BM25-style full-text search
      - Checkpoints use PG MVCC (multi-writer, no bottleneck)

    For DuckDB (fallback):
      - DuckDB file:   data/{account_id}.duckdb
      - LanceDB dir:   data/{account_id}.lance
      - Artifacts dir:  data/artifacts/{account_id}/
    """

    def __init__(self, config: ScalableConfig):
        self.config = config
        self._data_dir = Path(config.data_dir)
        self._artifacts_dir = Path(config.artifacts_dir)
        self._pg_manager = None

    async def initialize(self):
        """Initialize storage backend. Must be called before get_stores for postgres."""
        if self.config.storage_backend == StorageBackend.POSTGRES.value:
            from core.pg_store import PgStoreManager, PgConfig

            pg_config = PgConfig(
                dsn=self.config.pg_dsn,
                min_pool_size=self.config.pg_min_pool,
                max_pool_size=self.config.pg_max_pool,
                embedding_dim=self.config.embedding_dim,
                hnsw_m=self.config.pg_hnsw_m,
                hnsw_ef_construction=self.config.pg_hnsw_ef_construction,
                hnsw_ef_search=self.config.pg_hnsw_ef_search,
                distance_metric=self.config.pg_distance_metric,
                use_pg_search=self.config.pg_use_pg_search,
            )
            self._pg_manager = PgStoreManager(pg_config)
            await self._pg_manager.initialize()
            logger.info("PG storage backend initialized")

    async def get_stores(self, account_id: str) -> tuple:
        """
        Returns (store, vector_store_or_none, artifacts_path).

        For postgres: store is PgStore (handles both relational + vector).
                      vector_store is None (pgvector is built-in).
        For duckdb:   store is ClusteringStore, vector_store is LanceVectorStore.
        """
        artifacts_path = self._artifacts_dir / account_id
        artifacts_path.mkdir(parents=True, exist_ok=True)

        if self.config.storage_backend == StorageBackend.POSTGRES.value:
            if self._pg_manager is None:
                await self.initialize()
            store = await self._pg_manager.get_store(account_id)
            return store, None, artifacts_path

        # DuckDB fallback
        from core.store import ClusteringStore
        from core.lancedb_store import LanceVectorStore

        db_path = self._data_dir / f"{account_id}.duckdb"
        lance_path = self._data_dir / f"{account_id}.lance"
        store = ClusteringStore(db_path)
        lance_store = LanceVectorStore(lance_path)
        return store, lance_store, artifacts_path

    async def get_checkpoint_manager(self):
        """
        Returns checkpoint manager for configured backend.

        For postgres: PgCheckpointManager (MVCC multi-writer).
        For duckdb:   CheckpointManager (file-based, single-writer).
        """
        if self.config.storage_backend == StorageBackend.POSTGRES.value:
            if self._pg_manager is None:
                await self.initialize()
            return await self._pg_manager.get_checkpoint_manager()

        cp_path = self._data_dir / "pipeline_state"
        cp_path.mkdir(parents=True, exist_ok=True)
        return CheckpointManager(cp_path, backend=self.config.checkpoint_store)

    async def get_feedback_store(self, tenant_id: str):
        """Returns feedback store for configured backend."""
        if self.config.storage_backend == StorageBackend.POSTGRES.value:
            if self._pg_manager is None:
                await self.initialize()
            return await self._pg_manager.get_feedback_store(tenant_id)

        from core.feedback_store import FeedbackStore
        return FeedbackStore(tenant_id, data_dir=str(self._data_dir))

    async def close(self):
        """Close backend connections."""
        if self._pg_manager:
            await self._pg_manager.close()


# ══════════════════════════════════════════════════════════════════════════════
# §4  LLM CLIENT POOL — Rate-limited across accounts
# ══════════════════════════════════════════════════════════════════════════════


class LLMPool:
    """
    Global LLM connection pool with per-model semaphores.

    At 10K accounts hitting the same Azure OpenAI endpoint, we need
    a global rate limiter — not per-account. The pool enforces:
      - max_concurrent total LLM calls (across all accounts)
      - per-model deployment routing
      - retry with exponential backoff
      - call accounting for cost tracking
    """

    def __init__(self, max_concurrency: int = 8):
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._executor = None
        self._client = None
        self._call_count = 0
        self._total_tokens = 0
        self._total_latency = 0.0
        self._errors = 0

    @property
    def client(self):
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            )
        return self._client

    @property
    def executor(self):
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=16)
        return self._executor

    async def complete_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        model: str | None = None,
        retries: int = 3,
    ) -> dict | None:
        """Rate-limited async JSON completion with model routing."""
        deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            for attempt in range(retries):
                try:
                    t0 = time.time()
                    result = await loop.run_in_executor(
                        self.executor,
                        self._sync_call, prompt, temperature, max_tokens, deployment,
                    )
                    self._call_count += 1
                    self._total_latency += time.time() - t0
                    return result
                except json.JSONDecodeError:
                    logger.warning(f"JSON parse error (attempt {attempt + 1})")
                except Exception as e:
                    self._errors += 1
                    logger.warning(f"LLM error (attempt {attempt + 1}): {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(min(30, 1.5 ** (attempt + 1)))
        return None

    def _sync_call(
        self, prompt: str, temperature: float, max_tokens: int, deployment: str,
    ) -> dict:
        response = self.client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        usage = response.usage
        if usage:
            self._total_tokens += usage.total_tokens
        return json.loads(content)

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "total_latency_s": round(self._total_latency, 1),
            "avg_latency_s": round(self._total_latency / max(1, self._call_count), 2),
            "errors": self._errors,
        }

    def close(self):
        if self._executor:
            self._executor.shutdown(wait=False)


# ══════════════════════════════════════════════════════════════════════════════
# §5  EMBEDDING POOL — GPU-scheduled batching
# ══════════════════════════════════════════════════════════════════════════════


class EmbeddingPool:
    """
    Shared GPU embedding service with batched inference.

    At 1M docs (~80 chunks/doc = 80M chunks), embedding is the single
    largest GPU workload. This pool:
      - Queues embedding requests from all accounts
      - Batches them into GPU-optimal sizes (64-128)
      - Returns results via futures
      - Supports multiple GPUs via device_map
    """

    def __init__(self, config: ScalableConfig):
        self.config = config
        self._embedder = None
        self._lock = asyncio.Lock()

    @property
    def embedder(self):
        if self._embedder is None:
            from core.nomic_embedder import NomicEmbedder
            self._embedder = NomicEmbedder.get_instance(
                model_name=self.config.embedding_model,
                device=self.config.device,
            )
            self._embedder.warmup()
        return self._embedder

    def embed_sync(
        self,
        texts: list[str],
        prefix: str = "search_document",
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Synchronous embedding — for use inside executors."""
        bs = batch_size or self.config.embedding_batch_size
        return self.embedder.embed(texts, prefix=prefix, batch_size=bs)

    async def embed_async(
        self,
        texts: list[str],
        prefix: str = "search_document",
    ) -> np.ndarray:
        """Async embedding — offloads to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.embed_sync, texts, prefix,
        )

    def embed_single(self, text: str, prefix: str = "search_document") -> np.ndarray:
        return self.embedder.embed_single(text, prefix=prefix)


# ══════════════════════════════════════════════════════════════════════════════
# §6  PHASE 0 — FEATURE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════


class Phase0FeatureProcessor:
    """
    Phase 0: Document → Chunks → Embeddings → NL Index → Store.

    Embarrassingly parallel per document. At 1M docs, this runs as
    a pool of stateless workers processing micro-batches of 256 docs.

    Input:  Raw PDF paths or pre-parsed text
    Output: Chunks + embeddings persisted to store backend

    Storage backends:
      - postgres: chunks + embeddings stored as pgvector columns with
        HNSW index, tsvector GIN index auto-populated via GENERATED ALWAYS.
        No separate vector store needed — single INSERT populates both
        the relational row AND the vector/BM25 indexes.
      - duckdb: chunks in DuckDB + embeddings in LanceDB (legacy).

    Idempotency: If a document's chunks already exist in the store
    (by agreement_id), skip it. Use checkpoint manager for batch-level
    resume.
    """

    def __init__(
        self,
        config: ScalableConfig,
        embedding_pool: EmbeddingPool,
        checkpoint,  # CheckpointManager or PgCheckpointManager
    ):
        self.config = config
        self.embedding_pool = embedding_pool
        self.checkpoint = checkpoint
        self._is_pg = config.storage_backend == StorageBackend.POSTGRES.value

    async def _cp_phase_completed(self, phase: str, account_id: str) -> bool:
        """Backend-agnostic checkpoint phase check."""
        if self._is_pg:
            return await self.checkpoint.phase_completed(phase, account_id)
        return self.checkpoint.phase_completed(phase, account_id)

    async def _cp_is_completed(self, phase: str, account_id: str, unit_id: str) -> bool:
        if self._is_pg:
            return await self.checkpoint.is_completed(phase, account_id, unit_id)
        return self.checkpoint.is_completed(phase, account_id, unit_id)

    async def _cp_mark_started(self, phase: str, account_id: str, run_id: str, unit_id: str):
        if self._is_pg:
            await self.checkpoint.mark_started(phase, account_id, run_id, unit_id)
        else:
            self.checkpoint.mark_started(phase, account_id, run_id, unit_id)

    async def _cp_mark_completed(self, phase: str, account_id: str, run_id: str, unit_id: str, **kwargs):
        if self._is_pg:
            await self.checkpoint.mark_completed(phase, account_id, run_id, unit_id, **kwargs)
        else:
            self.checkpoint.mark_completed(phase, account_id, run_id, unit_id, **kwargs)

    async def _cp_mark_failed(self, phase: str, account_id: str, run_id: str, unit_id: str, error: str):
        if self._is_pg:
            await self.checkpoint.mark_failed(phase, account_id, run_id, unit_id, error)
        else:
            self.checkpoint.mark_failed(phase, account_id, run_id, unit_id, error)

    async def _cp_mark_phase_completed(self, phase: str, account_id: str, run_id: str, **kwargs):
        if self._is_pg:
            await self.checkpoint.mark_phase_completed(phase, account_id, run_id, **kwargs)
        else:
            self.checkpoint.mark_phase_completed(phase, account_id, run_id, **kwargs)

    async def process_account(
        self,
        account_id: str,
        run_id: str,
        pdf_paths: list[str | Path] | None = None,
        raw_docs: list[dict] | None = None,
        store=None,
        lance_store=None,
    ) -> dict:
        """
        Process all documents for one account.

        Args:
            account_id: Account identifier
            run_id: Pipeline run identifier
            pdf_paths: List of PDF file paths to parse
            raw_docs: Pre-parsed docs [{"agreement_id", "filename", "raw_text"}]
            store: PgStore or ClusteringStore
            lance_store: LanceVectorStore (None for postgres backend)

        Returns:
            Stats dict with n_docs, n_chunks, n_embeddings, timing_s
        """
        phase = Phase.FEATURE_PROCESSING.value
        if await self._cp_phase_completed(phase, account_id):
            logger.info(f"Phase 0 already completed for {account_id}, skipping")
            return {"skipped": True}

        t0 = time.time()

        # ── Step 0a: Ingest (parse PDFs → raw text) ──
        if raw_docs is None and pdf_paths:
            raw_docs = await self._parse_pdfs(pdf_paths)
        elif raw_docs is None:
            raise ValueError("Either pdf_paths or raw_docs must be provided")

        # Filter already-processed docs
        docs_to_process = []
        for doc in raw_docs:
            aid = doc["agreement_id"]
            if not await self._cp_is_completed(phase, account_id, aid):
                docs_to_process.append(doc)

        if not docs_to_process:
            logger.info(f"All {len(raw_docs)} docs already processed for {account_id}")
            await self._cp_mark_phase_completed(phase, account_id, run_id)
            return {"skipped": True, "n_docs": len(raw_docs)}

        logger.info(
            f"Phase 0: Processing {len(docs_to_process)}/{len(raw_docs)} "
            f"docs for {account_id}"
        )

        all_chunks = []
        all_embeddings = []

        # ── Step 0b + 0c: Chunk + Embed in micro-batches ──
        batch_size = self.config.max_docs_per_batch
        for batch_start in range(0, len(docs_to_process), batch_size):
            batch = docs_to_process[batch_start : batch_start + batch_size]
            batch_chunks, batch_embeddings = await self._process_doc_batch(
                batch, account_id, run_id,
            )
            all_chunks.extend(batch_chunks)
            if batch_embeddings is not None:
                all_embeddings.append(batch_embeddings)

        # ── Step 0d: NL Index (headings/sections) ──
        if self.config.build_nl_index:
            self._build_nl_index(all_chunks, store)

        # ── Step 0e: Persist to store ──
        if store and all_chunks:
            combined_emb = (
                np.vstack(all_embeddings) if len(all_embeddings) > 1
                else all_embeddings[0] if all_embeddings else None
            )
            await self._persist_phase0(
                store, lance_store, docs_to_process, all_chunks, combined_emb,
            )

        await self._cp_mark_phase_completed(phase, account_id, run_id, metadata={
            "n_docs": len(docs_to_process),
            "n_chunks": len(all_chunks),
        })

        return {
            "n_docs": len(docs_to_process),
            "n_chunks": len(all_chunks),
            "n_embeddings": sum(e.shape[0] for e in all_embeddings) if all_embeddings else 0,
            "timing_s": round(time.time() - t0, 2),
        }

    async def _persist_phase0(
        self,
        store,
        lance_store,
        docs: list[dict],
        chunks: list[dict],
        embeddings: np.ndarray | None,
    ):
        """
        Persist Phase 0 artifacts to configured backend.

        For postgres (PgStore):
          - Agreements via insert_agreement (async)
          - Chunks + embeddings via insert_chunks_batch (async, pgvector)
            → single INSERT populates: relational columns, vector HNSW index,
              AND tsvector GIN index (GENERATED column)
          - No separate vector store needed

        For duckdb (ClusteringStore + LanceVectorStore):
          - Agreements + chunks to DuckDB (sync)
          - Vectors to LanceDB (sync)
        """
        if self._is_pg:
            # ── Postgres path: async batch inserts ──
            for doc in docs:
                await store.insert_agreement(
                    agreement_id=doc["agreement_id"],
                    filename=doc.get("filename", ""),
                    raw_text=doc.get("raw_text", "")[:50000],
                )

            if chunks:
                await store.insert_chunks_batch(
                    chunk_ids=[c["chunk_id"] for c in chunks],
                    agreement_ids=[c["agreement_id"] for c in chunks],
                    texts=[c["chunk_text"] for c in chunks],
                    chunk_indexes=[c.get("chunk_index", 0) for c in chunks],
                    embeddings=embeddings,
                    token_counts=[c.get("token_count", 0) for c in chunks],
                )
                logger.info(
                    f"PG: persisted {len(chunks)} chunks with embeddings + "
                    f"HNSW index + tsvector BM25 index"
                )
        else:
            # ── DuckDB path: sync writes ──
            for doc in docs:
                store.insert_agreement(
                    agreement_id=doc["agreement_id"],
                    filename=doc.get("filename", ""),
                    raw_text=doc.get("raw_text", "")[:50000],
                )
            for chunk in chunks:
                store.insert_chunk(
                    chunk_id=chunk["chunk_id"],
                    agreement_id=chunk["agreement_id"],
                    chunk_text=chunk["chunk_text"],
                    chunk_index=chunk.get("chunk_index", 0),
                    token_count=chunk.get("token_count", 0),
                )

            # LanceDB vector store
            if lance_store and chunks and embeddings is not None:
                lance_store.upsert_chunks(
                    chunk_ids=[c["chunk_id"] for c in chunks],
                    vectors=embeddings,
                    texts=[c["chunk_text"] for c in chunks],
                    agreement_ids=[c["agreement_id"] for c in chunks],
                )

    async def _parse_pdfs(self, pdf_paths: list[str | Path]) -> list[dict]:
        """Parse PDFs in parallel. Returns list of doc dicts."""
        from core.pdf_parser import parse_pdf

        loop = asyncio.get_event_loop()
        sem = asyncio.Semaphore(self.config.pdf_parse_workers)

        async def _parse_one(path: str | Path):
            async with sem:
                return await loop.run_in_executor(None, parse_pdf, str(path))

        tasks = [_parse_one(p) for p in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"PDF parse failed: {r}")
                continue
            if r and len(r.get("raw_text", "")) > 100:
                docs.append(r)
        return docs

    async def _process_doc_batch(
        self,
        docs: list[dict],
        account_id: str,
        run_id: str,
    ) -> tuple[list[dict], np.ndarray | None]:
        """
        Process a micro-batch: chunk → embed → checkpoint each doc.
        """
        phase = Phase.FEATURE_PROCESSING.value
        all_chunks: list[dict] = []

        for doc in docs:
            aid = doc["agreement_id"]
            await self._cp_mark_started(phase, account_id, run_id, aid)

            try:
                chunks = self._chunk_document(doc)
                all_chunks.extend(chunks)
                await self._cp_mark_completed(phase, account_id, run_id, aid, metadata={
                    "n_chunks": len(chunks),
                })
            except Exception as e:
                logger.warning(f"Chunking failed for {aid}: {e}")
                await self._cp_mark_failed(phase, account_id, run_id, aid, str(e))

        if not all_chunks:
            return [], None

        # Batch-embed all chunks together (GPU-efficient)
        texts = [c["chunk_text"] for c in all_chunks]
        embeddings = await self.embedding_pool.embed_async(texts)

        return all_chunks, embeddings

    def _chunk_document(self, doc: dict) -> list[dict]:
        """Chunk one document using configured strategy."""
        from core.chunker import (
            semantic_chunk, sliding_window_chunk,
            _structural_clause_split, SemanticChunkerConfig,
        )

        text = doc.get("raw_text", "")
        aid = doc["agreement_id"]

        if self.config.chunk_strategy == "semantic":
            # Structural clause split first, then semantic sub-split
            chunk_config = SemanticChunkerConfig(
                min_chunk_chars=self.config.min_chunk_chars,
                max_chunk_chars=self.config.max_chunk_chars,
                clause_split=self.config.clause_split,
            )
            raw_chunks = semantic_chunk(
                text,
                embed_fn=self.embedding_pool.embed_sync,
                config=chunk_config,
            )
        else:
            raw_chunks = sliding_window_chunk(
                text,
                chunk_size=self.config.embedding_chunk_size,
                overlap=self.config.embedding_chunk_overlap,
            )

        # Annotate with metadata
        chunks = []
        for i, rc in enumerate(raw_chunks[:self.config.max_chunks_per_doc]):
            chunk_text = rc if isinstance(rc, str) else rc.get("text", rc.get("chunk_text", ""))
            chunks.append({
                "chunk_id": f"chk_{uuid.uuid4().hex[:12]}",
                "agreement_id": aid,
                "chunk_text": chunk_text,
                "chunk_index": i,
                "token_count": len(chunk_text) // 4,
                "chunk_type": "embedding",
            })
        return chunks

    def _build_nl_index(self, chunks: list[dict], store):
        """Build heading/section NL index from chunk texts."""
        import re
        heading_pattern = re.compile(
            r'^(?:#{1,4}\s+|(?:\d+\.)+\s+|Article\s+\d+|Section\s+\d+|ARTICLE\s+[IVXLC]+)',
            re.MULTILINE,
        )
        for chunk in chunks:
            text = chunk.get("chunk_text", "")
            if heading_pattern.search(text):
                # Store heading metadata alongside chunk (could be a separate table)
                chunk["has_heading"] = True


# ══════════════════════════════════════════════════════════════════════════════
# §7  PHASE 1 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════


class Phase1Clustering:
    """
    Phase 1: Embeddings → EVoC Clustering → Labels → Store.

    Per-account job. GPU-accelerated for the Optuna trial loop.
    10K accounts = 10K clustering jobs (queued, not simultaneous).

    Input:  Chunk embeddings from Phase 0 (DuckDB + LanceDB)
    Output: Cluster assignments, centroids, labels in DuckDB

    Scaling:
      - Optuna trials run sequentially (EVoC is fast: ~5s/trial)
      - 30 trials × 5s = ~2.5 min per account
      - 10K accounts × 2.5 min = ~17 hours (serial) or ~4h (4x GPU parallel)
      - For 1M docs: document-level means ~100K embeddings per large account
        → EVoC handles this in one pass (no batching needed)
    """

    def __init__(
        self,
        config: ScalableConfig,
        embedding_pool: EmbeddingPool,
        llm_pool: LLMPool,
        checkpoint: CheckpointManager,
    ):
        self.config = config
        self.embedding_pool = embedding_pool
        self.llm_pool = llm_pool
        self.checkpoint = checkpoint

    async def cluster_account(
        self,
        account_id: str,
        run_id: str,
        store=None,
        lance_store=None,
    ) -> dict:
        """
        Run full clustering pipeline for one account.

        Steps:
          1a. Load embeddings from store
          1b. EVoC + Optuna optimization (30 trials)
          1c. Best-layer selection (silhouette + cosine scoring)
          1d. Near-duplicate cluster merge
          1e. Outlier recovery (iterative re-cluster / soft assign)
          1f. LLM label synthesis
          1g. Persist clusters + assignments

        Returns: Stats dict
        """
        phase = Phase.CLUSTERING.value
        if self.checkpoint.phase_completed(phase, account_id):
            logger.info(f"Phase 1 already completed for {account_id}, skipping")
            return {"skipped": True}

        t0 = time.time()
        meta: dict[str, Any] = {"account_id": account_id}

        # ── 1a. Load embeddings ──
        t_load = time.time()
        chunk_ids, embeddings = store.get_all_chunk_embeddings()
        if embeddings is None or len(embeddings) == 0:
            # Fall back to LanceDB if DuckDB embeddings not stored
            logger.info("Loading embeddings from vector store...")
            chunks = store.get_chunks()
            if not chunks:
                return {"error": "no_chunks", "account_id": account_id}

            texts = [c["chunk_text"] for c in chunks]
            embeddings = self.embedding_pool.embed_sync(texts)
            chunk_ids = [c["chunk_id"] for c in chunks]

        n_chunks = len(chunk_ids)
        meta["n_chunks"] = n_chunks
        meta["load_time_s"] = round(time.time() - t_load, 2)
        logger.info(f"Phase 1: Clustering {n_chunks} chunks for {account_id}")

        if n_chunks < self.config.min_clusters:
            logger.warning(f"Too few chunks ({n_chunks}) for clustering")
            return {"error": "too_few_chunks", **meta}

        # ── 1b. EVoC + Optuna ──
        t_cluster = time.time()
        best_params, labels, best_score = await self._run_evoc_optuna(embeddings)
        meta["evoc_time_s"] = round(time.time() - t_cluster, 2)
        meta["best_params"] = best_params
        meta["best_score"] = round(best_score, 4)

        # ── 1c. Extract cluster structure ──
        unique_labels = set(labels)
        unique_labels.discard(-1)
        n_clusters = len(unique_labels)
        n_outliers = int(np.sum(labels == -1))
        meta["n_clusters_raw"] = n_clusters
        meta["n_outliers_raw"] = n_outliers
        meta["outlier_ratio_raw"] = round(n_outliers / max(n_chunks, 1), 3)

        # ── 1d. Near-duplicate merge ──
        t_merge = time.time()
        labels, merge_map = self._merge_near_duplicates(embeddings, labels)
        meta["merge_time_s"] = round(time.time() - t_merge, 2)
        meta["merges_performed"] = len(merge_map)

        # ── 1e. Outlier recovery ──
        t_outlier = time.time()
        labels = await self._recover_outliers(embeddings, labels)
        n_outliers_post = int(np.sum(labels == -1))
        meta["outlier_recovery_time_s"] = round(time.time() - t_outlier, 2)
        meta["n_outliers_post"] = n_outliers_post
        meta["outlier_ratio_post"] = round(n_outliers_post / max(n_chunks, 1), 3)

        # ── 1f. Compute centroids + quality ──
        cluster_records = self._build_cluster_records(
            embeddings, labels, chunk_ids,
        )

        # ── 1g. LLM label synthesis ──
        t_label = time.time()
        chunks_by_cluster = self._group_chunks_by_cluster(chunk_ids, labels, store)
        cluster_records = await self._synthesize_labels(
            cluster_records, chunks_by_cluster,
        )
        meta["label_time_s"] = round(time.time() - t_label, 2)

        # ── 1h. Persist ──
        t_persist = time.time()
        self._persist_clusters(store, cluster_records, chunk_ids, labels, embeddings)
        meta["persist_time_s"] = round(time.time() - t_persist, 2)

        meta["n_clusters_final"] = len(cluster_records)
        meta["total_time_s"] = round(time.time() - t0, 2)

        self.checkpoint.mark_phase_completed(phase, account_id, run_id, metadata=meta)
        return meta

    async def _run_evoc_optuna(self, embeddings: np.ndarray) -> tuple[dict, np.ndarray, float]:
        """Run EVoC clustering with Optuna TPE hyperparameter optimization."""
        from core.evoc_pipeline import evoc_optimize

        loop = asyncio.get_event_loop()

        def _optimize():
            return evoc_optimize(
                embeddings,
                n_trials=self.config.optuna_trials,
                param_ranges={
                    "base_min_cluster_size": self.config.evoc_min_cluster_size_range,
                    "noise_level": self.config.evoc_noise_level_range,
                    "n_neighbors": self.config.evoc_n_neighbors_range,
                    "min_samples": self.config.evoc_min_samples_range,
                },
            )

        best_params, best_labels, best_score = await loop.run_in_executor(
            None, _optimize,
        )
        return best_params, best_labels, best_score

    def _merge_near_duplicates(
        self, embeddings: np.ndarray, labels: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Merge clusters with centroid cosine sim > merge_threshold."""
        from core.topic_merger import HierarchicalTopicMerger

        # Build cluster centroids
        unique_labels = sorted(set(labels) - {-1})
        if len(unique_labels) < 2:
            return labels, {}

        centroids = {}
        for lbl in unique_labels:
            mask = labels == lbl
            centroid = embeddings[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            centroids[lbl] = centroid

        # Pairwise cosine similarity
        merge_map: dict[int, int] = {}  # old_label → keep_label
        merged = set()
        label_list = list(centroids.keys())

        for i, lbl_a in enumerate(label_list):
            if lbl_a in merged:
                continue
            for lbl_b in label_list[i + 1:]:
                if lbl_b in merged:
                    continue
                sim = float(np.dot(centroids[lbl_a], centroids[lbl_b]))
                if sim >= self.config.merge_threshold:
                    merge_map[lbl_b] = lbl_a
                    merged.add(lbl_b)

        # Apply merges
        new_labels = labels.copy()
        for old, new in merge_map.items():
            new_labels[labels == old] = new

        return new_labels, merge_map

    async def _recover_outliers(
        self, embeddings: np.ndarray, labels: np.ndarray,
    ) -> np.ndarray:
        """Recover outliers using configured strategy."""
        outlier_mask = labels == -1
        n_outliers = int(outlier_mask.sum())
        if n_outliers == 0:
            return labels

        outlier_ratio = n_outliers / len(labels)
        if outlier_ratio <= self.config.max_outlier_ratio:
            return self._assign_outliers_nearest(embeddings, labels)

        strategy = self.config.outlier_strategy

        if strategy == "iterative":
            return await self._recover_outliers_iterative(embeddings, labels)
        elif strategy == "soft":
            return self._recover_outliers_soft(embeddings, labels)
        else:
            return self._assign_outliers_nearest(embeddings, labels)

    def _assign_outliers_nearest(
        self, embeddings: np.ndarray, labels: np.ndarray,
    ) -> np.ndarray:
        """Assign each outlier to its nearest cluster centroid."""
        result = labels.copy()
        unique = sorted(set(labels) - {-1})
        if not unique:
            return result

        centroids = np.array([
            embeddings[labels == lbl].mean(axis=0) for lbl in unique
        ])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1
        centroids = centroids / norms

        outlier_indices = np.where(labels == -1)[0]
        if len(outlier_indices) == 0:
            return result

        outlier_embs = embeddings[outlier_indices]
        outlier_norms = np.linalg.norm(outlier_embs, axis=1, keepdims=True)
        outlier_norms[outlier_norms == 0] = 1
        outlier_embs = outlier_embs / outlier_norms

        sims = outlier_embs @ centroids.T  # (n_outliers, n_clusters)

        for i, idx in enumerate(outlier_indices):
            best_cluster_pos = int(np.argmax(sims[i]))
            best_sim = float(sims[i, best_cluster_pos])
            if best_sim >= self.config.outlier_min_similarity:
                result[idx] = unique[best_cluster_pos]
            # else leave as -1 (unclassifiable)

        return result

    async def _recover_outliers_iterative(
        self, embeddings: np.ndarray, labels: np.ndarray,
    ) -> np.ndarray:
        """Re-cluster the outlier pool at lower min_cluster_size."""
        from core.evoc_pipeline import evoc_cluster

        result = labels.copy()
        outlier_indices = np.where(labels == -1)[0]
        if len(outlier_indices) < 3:
            return self._assign_outliers_nearest(embeddings, result)

        outlier_embs = embeddings[outlier_indices]
        loop = asyncio.get_event_loop()

        def _recluster():
            return evoc_cluster(
                outlier_embs,
                base_min_cluster_size=max(2, self.config.evoc_min_cluster_size_range[0] - 1),
                noise_level=self.config.evoc_noise_level_range[0],
            )

        sub_labels, _, _ = await loop.run_in_executor(None, _recluster)

        # Map sub-labels to new global labels
        max_existing = max(set(labels) - {-1}) if set(labels) - {-1} else 0
        for i, idx in enumerate(outlier_indices):
            if sub_labels[i] != -1:
                result[idx] = max_existing + 1 + sub_labels[i]

        # Remaining outliers → nearest assignment
        return self._assign_outliers_nearest(embeddings, result)

    def _recover_outliers_soft(
        self, embeddings: np.ndarray, labels: np.ndarray,
    ) -> np.ndarray:
        """Soft assignment: outliers get top-1 cluster (for now)."""
        return self._assign_outliers_nearest(embeddings, labels)

    def _build_cluster_records(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        chunk_ids: list[str],
    ) -> list[dict]:
        """Build cluster records with centroids and quality scores."""
        unique = sorted(set(labels) - {-1})
        records = []
        for lbl in unique:
            mask = labels == lbl
            cluster_embs = embeddings[mask]
            centroid = cluster_embs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            # Intra-cluster cosine similarity
            normed = cluster_embs / np.linalg.norm(
                cluster_embs, axis=1, keepdims=True
            ).clip(1e-10)
            sims = normed @ centroid
            quality = float(np.mean(sims))

            records.append({
                "cluster_id": f"cls_{uuid.uuid4().hex[:10]}",
                "label_id": int(lbl),
                "label": f"Cluster_{lbl}",  # placeholder, synthesized later
                "description": "",
                "keywords": [],
                "chunk_count": int(mask.sum()),
                "agreement_count": len(set(
                    chunk_ids[i] for i in np.where(mask)[0]
                )),
                "quality_score": round(quality, 4),
                "centroid": centroid,
            })
        return records

    def _group_chunks_by_cluster(
        self, chunk_ids: list[str], labels: np.ndarray, store,
    ) -> dict[str, list[dict]]:
        """Group chunk texts by cluster for label synthesis."""
        all_chunks = {c["chunk_id"]: c for c in store.get_chunks()}
        clusters: dict[str, list[dict]] = {}
        for i, cid in enumerate(chunk_ids):
            lbl = int(labels[i])
            if lbl == -1:
                continue
            key = str(lbl)
            if key not in clusters:
                clusters[key] = []
            chunk = all_chunks.get(cid)
            if chunk:
                clusters[key].append(chunk)
        return clusters

    async def _synthesize_labels(
        self,
        cluster_records: list[dict],
        chunks_by_cluster: dict[str, list[dict]],
    ) -> list[dict]:
        """LLM-driven label synthesis for each cluster."""
        sem = asyncio.Semaphore(4)

        async def _label_one(record: dict) -> dict:
            async with sem:
                key = str(record["label_id"])
                chunks = chunks_by_cluster.get(key, [])
                if not chunks:
                    return record

                sample_texts = "\n---\n".join(
                    c.get("chunk_text", "")[:500] for c in chunks[:5]
                )
                prompt = (
                    f"Analyze these {len(chunks)} clause excerpts from legal agreements.\n\n"
                    f"SAMPLE EXCERPTS:\n{sample_texts}\n\n"
                    f"Return JSON:\n"
                    f'{{"label": "<2-4 word clause type name>", '
                    f'"description": "<one sentence description>", '
                    f'"keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"]}}'
                )
                result = await self.llm_pool.complete_json(
                    prompt, temperature=0.1, max_tokens=300,
                    model=self.config.label_synthesis_model,
                )
                if result:
                    record["label"] = result.get("label", record["label"])
                    record["description"] = result.get("description", "")
                    record["keywords"] = result.get("keywords", [])
                return record

        tasks = [_label_one(r) for r in cluster_records]
        return await asyncio.gather(*tasks)

    def _persist_clusters(
        self, store, cluster_records: list[dict],
        chunk_ids: list[str], labels: np.ndarray, embeddings: np.ndarray,
    ):
        """Persist cluster records and assignments to DuckDB."""
        for record in cluster_records:
            store.save_cluster(
                cluster_id=record["cluster_id"],
                label=record["label"],
                description=record.get("description", ""),
                keywords=json.dumps(record.get("keywords", [])),
                chunk_count=record["chunk_count"],
            )
            centroid = record.get("centroid")
            if centroid is not None:
                store.save_centroid(
                    cluster_id=record["cluster_id"],
                    centroid=centroid,
                    chunk_count=record["chunk_count"],
                )

        # Build label_id → cluster_id map
        id_map = {r["label_id"]: r["cluster_id"] for r in cluster_records}

        for i, cid in enumerate(chunk_ids):
            lbl = int(labels[i])
            cluster_id = id_map.get(lbl)
            if cluster_id:
                store.assign_chunk_to_cluster(
                    chunk_id=cid,
                    cluster_id=cluster_id,
                    probability=1.0,
                    is_outlier=False,
                )
            elif lbl == -1:
                store.assign_chunk_to_cluster(
                    chunk_id=cid,
                    cluster_id="unclassified",
                    probability=0.0,
                    is_outlier=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# §8  PHASE 2 — DISCOVERY + EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════


class Phase2Extraction:
    """
    Phase 2: Clusters → Field Discovery → Extraction → Store.

    Per-cluster work, bounded by LLM concurrency. This is the most
    LLM-intensive phase — at 10K accounts with ~50 clusters each,
    that's 500K cluster-level discovery jobs + millions of extractions.

    Input:  Clusters + chunks from Phase 1
    Output: Field definitions + extracted values in DuckDB

    Scaling:
      - Discovery: ~25-40s/cluster (RLM V3), bounded by LLM rate limits
      - Extraction: ~2-5s/doc/field (RAG), bounded by LLM concurrency
      - 50 clusters/account × 10 docs/cluster × 15 fields = 7,500 LLM calls/account
      - 10K accounts × 7,500 = 75M LLM calls total → need aggressive batching + caching
    """

    def __init__(
        self,
        config: ScalableConfig,
        embedding_pool: EmbeddingPool,
        llm_pool: LLMPool,
        checkpoint: CheckpointManager,
    ):
        self.config = config
        self.embedding_pool = embedding_pool
        self.llm_pool = llm_pool
        self.checkpoint = checkpoint

    async def extract_account(
        self,
        account_id: str,
        run_id: str,
        store=None,
        lance_store=None,
    ) -> dict:
        """
        Run full discovery + extraction for one account.

        Steps:
          2a. Load clusters from store
          2b. Per-cluster RLM V3 field discovery (with inline extraction)
          2c. Cross-cluster field deduplication
          2d. RAG extraction on remaining (doc, field) pairs
          2e. Normalize values + persist

        Returns: Stats dict
        """
        phase = Phase.EXTRACTION.value
        if self.checkpoint.phase_completed(phase, account_id):
            logger.info(f"Phase 2 already completed for {account_id}, skipping")
            return {"skipped": True}

        t0 = time.time()
        meta: dict[str, Any] = {"account_id": account_id}

        # ── 2a. Load clusters ──
        clusters = store.get_clusters()
        if not clusters:
            return {"error": "no_clusters", **meta}

        meta["n_clusters"] = len(clusters)
        logger.info(f"Phase 2: Processing {len(clusters)} clusters for {account_id}")

        # ── 2b. Discovery (per-cluster, bounded concurrency) ──
        t_discovery = time.time()
        cluster_sem = asyncio.Semaphore(self.config.max_cluster_concurrency)
        all_fields: list[dict] = []
        all_inline_extractions: list[dict] = []
        discovery_metas: dict[str, dict] = {}

        async def _discover_one(cluster: dict):
            async with cluster_sem:
                cid = cluster["cluster_id"]
                if self.checkpoint.is_completed(phase, account_id, f"discover_{cid}"):
                    return

                self.checkpoint.mark_started(phase, account_id, run_id, f"discover_{cid}")
                try:
                    fields, inline_ext, disc_meta = await self._discover_cluster(
                        cluster, store,
                    )
                    all_fields.extend(fields)
                    all_inline_extractions.extend(inline_ext)
                    discovery_metas[cid] = disc_meta

                    # Persist field definitions immediately
                    for f in fields:
                        field_id = f"fld_{uuid.uuid4().hex[:10]}"
                        f["field_id"] = field_id
                        store.upsert_field(
                            field_id=field_id,
                            cluster_id=cid,
                            name=f["name"],
                            field_type=f.get("type", "text"),
                            description=f.get("description", ""),
                            source=f.get("source", "rlm_v3"),
                            examples=f.get("examples", []),
                        )

                    # Persist inline extractions
                    for ext in inline_ext:
                        field_id = next(
                            (f["field_id"] for f in fields if f["name"] == ext["field_name"]),
                            None,
                        )
                        if field_id:
                            store.insert_extraction(
                                extraction_id=f"ext_{uuid.uuid4().hex[:10]}",
                                agreement_id=ext.get("agreement_id", ""),
                                field_id=field_id,
                                value=ext["value"],
                                confidence=ext.get("confidence", 0.5),
                                source_chunk_id=ext.get("chunk_id", ""),
                            )

                    self.checkpoint.mark_completed(
                        phase, account_id, run_id, f"discover_{cid}",
                        metadata={"n_fields": len(fields), "n_inline": len(inline_ext)},
                    )
                except Exception as e:
                    logger.warning(f"Discovery failed for cluster {cid}: {e}")
                    self.checkpoint.mark_failed(
                        phase, account_id, run_id, f"discover_{cid}", str(e),
                    )

        discovery_tasks = [_discover_one(c) for c in clusters]
        await asyncio.gather(*discovery_tasks)

        meta["discovery_time_s"] = round(time.time() - t_discovery, 2)
        meta["n_fields_raw"] = len(all_fields)
        meta["n_inline_extractions"] = len(all_inline_extractions)

        # ── 2c. Cross-cluster field deduplication ──
        t_dedup = time.time()
        n_before = len(all_fields)
        self._deduplicate_fields(store)
        meta["dedup_time_s"] = round(time.time() - t_dedup, 2)

        # ── 2d. RAG Extraction on remaining docs ──
        if self.config.extraction_backend == ExtractionBackend.RAG.value:
            t_extract = time.time()
            extract_stats = await self._run_rag_extraction(
                clusters, store, lance_store, discovery_metas,
            )
            meta["rag_extraction_time_s"] = round(time.time() - t_extract, 2)
            meta.update(extract_stats)
        elif self.config.extraction_backend == ExtractionBackend.SEQUENTIAL.value:
            t_extract = time.time()
            extract_stats = await self._run_sequential_extraction(
                clusters, store, discovery_metas,
            )
            meta["sequential_extraction_time_s"] = round(time.time() - t_extract, 2)
            meta.update(extract_stats)

        meta["total_time_s"] = round(time.time() - t0, 2)
        self.checkpoint.mark_phase_completed(phase, account_id, run_id, metadata=meta)
        return meta

    async def _discover_cluster(
        self,
        cluster: dict,
        store,
    ) -> tuple[list[dict], list[dict], dict]:
        """
        Run RLM V3 discovery (+ inline extraction if unified mode).

        Returns: (fields, inline_extractions, meta)
        """
        cid = cluster["cluster_id"]
        label = cluster.get("label", "Unknown")
        keywords = cluster.get("keywords", [])
        if isinstance(keywords, str):
            try:
                keywords = json.loads(keywords)
            except (json.JSONDecodeError, TypeError):
                keywords = []

        chunk_rows = store.get_chunks(cluster_id=cid)
        if not chunk_rows:
            return [], [], {"error": "no_chunks"}

        chunk_texts = [r["chunk_text"] for r in chunk_rows]
        chunk_ids = [r["chunk_id"] for r in chunk_rows]
        agreement_ids = [r["agreement_id"] for r in chunk_rows]

        loop = asyncio.get_event_loop()

        if self.config.discovery_mode in (
            DiscoveryMode.RLM_V3_UNIFIED.value,
            "rlm_v3_unified",
        ):
            def _run():
                from core.rlm_v3 import RLMV3, RLMV3Config
                v3_config = RLMV3Config(
                    max_iterations=self.config.rlm_max_iterations,
                    max_llm_calls=self.config.rlm_max_llm_calls,
                    max_sample_chunks=self.config.rlm_max_sample_chunks,
                    max_fields=self.config.max_fields_per_cluster,
                    extract_inline=True,
                    min_extraction_confidence=self.config.min_extraction_confidence,
                    briefing_max_tokens=self.config.rlm_briefing_max_tokens,
                )
                v3 = RLMV3(config=v3_config)
                return v3.discover_and_extract(
                    label, keywords, chunk_texts,
                    chunk_ids=chunk_ids, agreement_ids=agreement_ids,
                )

            try:
                fields, extractions, disc_meta = await loop.run_in_executor(None, _run)
                return fields, extractions, disc_meta
            except Exception as e:
                logger.warning(f"V3 unified failed for {cid}, falling back to discovery-only: {e}")

        # Fallback: discovery-only
        def _run_discover():
            from core.rlm_v3 import RLMV3, RLMV3Config
            v3_config = RLMV3Config(
                max_iterations=self.config.rlm_max_iterations,
                max_llm_calls=self.config.rlm_max_llm_calls,
                max_sample_chunks=self.config.rlm_max_sample_chunks,
                max_fields=self.config.max_fields_per_cluster,
                briefing_max_tokens=self.config.rlm_briefing_max_tokens,
            )
            v3 = RLMV3(config=v3_config)
            return v3.discover(label, keywords, chunk_texts)

        try:
            fields, disc_meta = await loop.run_in_executor(None, _run_discover)
            return fields, [], disc_meta
        except Exception as e:
            logger.warning(f"V3 discovery failed for {cid}: {e}")
            return [], [], {"error": str(e)}

    def _deduplicate_fields(self, store):
        """Cross-cluster field deduplication using embedding similarity."""
        all_fields = store.get_fields()
        if len(all_fields) < 2:
            return

        # Build field descriptions for embedding
        descriptions = []
        for f in all_fields:
            desc = f"{f['name'].replace('_', ' ')} — {f.get('description', '')}"
            descriptions.append(desc)

        embeddings = self.embedding_pool.embed_sync(descriptions)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = embeddings / norms

        # Pairwise cosine → find duplicates
        sims = normed @ normed.T
        merged = set()

        for i in range(len(all_fields)):
            if i in merged:
                continue
            for j in range(i + 1, len(all_fields)):
                if j in merged:
                    continue
                if sims[i, j] >= self.config.field_dedup_threshold:
                    # Keep the field with more examples or higher confidence
                    keep = all_fields[i]
                    drop = all_fields[j]
                    merged.add(j)
                    logger.debug(
                        f"Dedup: merging '{drop['name']}' into '{keep['name']}' "
                        f"(sim={sims[i,j]:.3f})"
                    )

        logger.info(f"Field dedup: {len(merged)} fields merged from {len(all_fields)}")

    async def _run_rag_extraction(
        self,
        clusters: list[dict],
        store,
        lance_store,
        discovery_metas: dict[str, dict],
    ) -> dict:
        """Run RAG extraction across all clusters."""
        from core.rag_extractor import RAGExtractor, RAGConfig

        rag_config = RAGConfig(
            cochran_z=self.config.cochran_z,
            cochran_e=self.config.cochran_e,
            max_sample_docs=self.config.max_sample_docs,
            max_retry_rounds=self.config.rag_max_retry_rounds,
            enable_reviewer=self.config.rag_enable_reviewer,
            semantic_top_k=self.config.rag_semantic_top_k,
            final_top_k=self.config.rag_final_top_k,
            complex_model=self.config.complex_model,
            simple_model=self.config.simple_model,
            min_extraction_confidence=self.config.min_extraction_confidence,
            max_concurrent_docs=self.config.max_doc_concurrency,
            max_concurrent_fields=self.config.max_field_concurrency,
        )

        extractor = RAGExtractor(
            store=store,
            llm=self.llm_pool,
            embed_fn=self.embedding_pool.embed_sync,
            config=rag_config,
            lance_store=lance_store,
        )

        total_extractions = 0
        total_retries = 0
        total_rejections = 0

        cluster_sem = asyncio.Semaphore(self.config.max_cluster_concurrency)

        async def _extract_one(cluster: dict):
            nonlocal total_extractions, total_retries, total_rejections
            async with cluster_sem:
                cid = cluster["cluster_id"]
                fields = store.get_fields(cluster_id=cid)
                if not fields:
                    return

                label = cluster.get("label", "Unknown")
                keywords = cluster.get("keywords", [])

                # Build context from discovery meta
                context = ""
                disc_meta = discovery_metas.get(cid, {})
                clause_types = disc_meta.get("clause_types", [])
                if clause_types:
                    ct_str = "; ".join(
                        f"{ct.get('name', '')}" for ct in clause_types[:3]
                    )
                    context += f"Clause sub-types: {ct_str}\n"

                try:
                    batch = await extractor.extract_cluster(
                        cluster_id=cid,
                        fields=fields,
                        cluster_label=label,
                        cluster_keywords=keywords,
                        context_section=context,
                    )
                    total_extractions += batch.meta.get("n_persisted", 0)
                    total_retries += batch.meta.get("total_retries", 0)
                    total_rejections += batch.meta.get("total_reviewer_rejections", 0)
                except Exception as e:
                    logger.warning(f"RAG extraction failed for {cid}: {e}")

        tasks = [_extract_one(c) for c in clusters]
        await asyncio.gather(*tasks)

        return {
            "total_extractions": total_extractions,
            "total_retries": total_retries,
            "total_reviewer_rejections": total_rejections,
        }

    async def _run_sequential_extraction(
        self,
        clusters: list[dict],
        store,
        discovery_metas: dict[str, dict],
    ) -> dict:
        """
        Sequential chunk extraction — processes ALL chunks in order,
        with knowledge carryforward between chunks.

        Placeholder for Pipeline 3 implementation.
        """
        logger.info("Sequential extraction: not yet implemented, falling back to basic per-chunk")
        # TODO: Implement sequential extraction with carryforward + judge LLM
        # For now, return empty — the inline extractions from Phase 2a cover sampled chunks
        return {"sequential_status": "not_implemented", "total_extractions": 0}


# ══════════════════════════════════════════════════════════════════════════════
# §9  ORCHESTRATOR — Runs all 3 phases for N accounts
# ══════════════════════════════════════════════════════════════════════════════


class ScalablePipeline:
    """
    Top-level orchestrator: runs Phase 0 → 1 → 2 for multiple accounts.

    Scaling model:
    ─────────────────────────────────────────────────────────────────
    1M docs across 10K accounts → avg 100 docs/account

    Phase 0 (Feature Processing):
      - 100 docs × 80 chunks = 8,000 chunks/account
      - 8,000 × 768-dim × 4 bytes = ~24 MB embeddings/account
      - GPU embedding: ~50 chunks/sec on A100 → ~160s/account
      - 10K accounts: ~18 days serial, ~18 hours @ 24x GPU parallel
      - Strategy: Batch across accounts, share GPU embedding pool

    Phase 1 (Clustering):
      - 8,000 chunks per EVoC run → <10s clustering
      - 30 Optuna trials × 10s = ~5 min/account
      - 10K accounts: ~35 days serial, ~4 days @ 8x parallel
      - Strategy: Queue accounts, 1 GPU per EVoC run

    Phase 2 (Discovery + Extraction):
      - ~50 clusters/account × 30s discovery = ~25 min/account
      - ~50 clusters × 10 docs × 15 fields × 0.5s = ~1 hour/account
      - 10K accounts: bottleneck is LLM rate limiting
      - At 1000 TPM: ~75K calls/account, 10K accounts = 750M calls
      - Strategy: Semaphore-gated LLM pool, priority queuing
    ─────────────────────────────────────────────────────────────────

    Usage:
        config = ScalableConfig(device="cuda", optuna_trials=30)
        pipeline = ScalablePipeline(config)
        stats = await pipeline.run(account_ids=["acct_001", "acct_002"])
    """

    def __init__(self, config: ScalableConfig | None = None):
        self.config = config or ScalableConfig()
        self.store_factory = StoreFactory(self.config)
        self.embedding_pool = EmbeddingPool(self.config)
        self.llm_pool = LLMPool(max_concurrency=self.config.max_llm_concurrency)
        self._initialized = False

    async def _ensure_initialized(self):
        """Lazy-init: create pool + checkpoint manager on first use."""
        if self._initialized:
            return
        await self.store_factory.initialize()
        self.checkpoint = await self.store_factory.get_checkpoint_manager()
        self.phase0 = Phase0FeatureProcessor(
            self.config, self.embedding_pool, self.checkpoint,
        )
        self.phase1 = Phase1Clustering(
            self.config, self.embedding_pool, self.llm_pool, self.checkpoint,
        )
        self.phase2 = Phase2Extraction(
            self.config, self.embedding_pool, self.llm_pool, self.checkpoint,
        )
        self._initialized = True

    async def run(
        self,
        account_ids: list[str],
        pdf_dirs: dict[str, str] | None = None,
        raw_docs_map: dict[str, list[dict]] | None = None,
        phases: list[str] | None = None,
    ) -> dict:
        """
        Run the full pipeline for multiple accounts.

        Args:
            account_ids: List of account identifiers
            pdf_dirs: Map of account_id → PDF directory path
            raw_docs_map: Map of account_id → pre-parsed doc list
            phases: Which phases to run ["phase0", "phase1", "phase2"]
                    Default: all phases

        Returns:
            Combined stats for all accounts
        """
        await self._ensure_initialized()
        phases = phases or [p.value for p in Phase]
        t0 = time.time()
        results: dict[str, dict] = {}

        for account_id in account_ids:
            run_id = f"run_{uuid.uuid4().hex[:8]}"
            logger.info(f"{'='*60}")
            logger.info(f"Pipeline: account={account_id}, run={run_id}")
            logger.info(f"{'='*60}")

            store, lance_store, artifacts = await self.store_factory.get_stores(account_id)
            account_result: dict[str, Any] = {"run_id": run_id}

            try:
                # ── Phase 0: Feature Processing ──
                if Phase.FEATURE_PROCESSING.value in phases:
                    pdf_dir = (pdf_dirs or {}).get(account_id)
                    raw_docs = (raw_docs_map or {}).get(account_id)
                    if pdf_dir or raw_docs:
                        p0_result = await self.phase0.process_account(
                            account_id=account_id,
                            run_id=run_id,
                            pdf_paths=list(Path(pdf_dir).glob("*.pdf")) if pdf_dir else None,
                            raw_docs=raw_docs,
                            store=store,
                            lance_store=lance_store,
                        )
                        account_result["phase0"] = p0_result
                    else:
                        logger.info(f"Skipping Phase 0 for {account_id} (no docs provided)")

                # ── Phase 1: Clustering ──
                if Phase.CLUSTERING.value in phases:
                    p1_result = await self.phase1.cluster_account(
                        account_id=account_id,
                        run_id=run_id,
                        store=store,
                        lance_store=lance_store,
                    )
                    account_result["phase1"] = p1_result

                # ── Phase 2: Discovery + Extraction ──
                if Phase.EXTRACTION.value in phases:
                    p2_result = await self.phase2.extract_account(
                        account_id=account_id,
                        run_id=run_id,
                        store=store,
                        lance_store=lance_store,
                    )
                    account_result["phase2"] = p2_result

            except Exception as e:
                logger.error(f"Pipeline failed for {account_id}: {e}")
                account_result["error"] = str(e)

            results[account_id] = account_result

        total_time = round(time.time() - t0, 2)
        return {
            "accounts": results,
            "n_accounts": len(account_ids),
            "total_time_s": total_time,
            "llm_stats": self.llm_pool.stats,
        }

    async def run_phase0(
        self, account_id: str, pdf_paths: list[str] | None = None,
        raw_docs: list[dict] | None = None,
    ) -> dict:
        """Run only Phase 0 for a single account."""
        await self._ensure_initialized()
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        store, lance_store, _ = await self.store_factory.get_stores(account_id)
        return await self.phase0.process_account(
            account_id, run_id, pdf_paths=pdf_paths,
            raw_docs=raw_docs, store=store, lance_store=lance_store,
        )

    async def run_phase1(self, account_id: str) -> dict:
        """Run only Phase 1 for a single account."""
        await self._ensure_initialized()
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        store, lance_store, _ = await self.store_factory.get_stores(account_id)
        return await self.phase1.cluster_account(
            account_id, run_id, store=store, lance_store=lance_store,
        )

    async def run_phase2(self, account_id: str) -> dict:
        """Run only Phase 2 for a single account."""
        await self._ensure_initialized()
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        store, lance_store, _ = await self.store_factory.get_stores(account_id)
        return await self.phase2.extract_account(
            account_id, run_id, store=store, lance_store=lance_store,
        )

    async def close(self):
        self.llm_pool.close()
        if hasattr(self, 'checkpoint') and self.checkpoint:
            self.checkpoint.close()
        await self.store_factory.close()


# ══════════════════════════════════════════════════════════════════════════════
# §10  CLI
# ══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Scalable 3-Phase Pipeline: Feature Processing → Clustering → Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline for one account
  python -m core.scalable_pipeline --accounts acct_001 --pdf-dir /pdfs/acct_001

  # Only clustering (Phase 1) — assumes Phase 0 already ran
  python -m core.scalable_pipeline --accounts acct_001 --phase phase1

  # Multiple accounts
  python -m core.scalable_pipeline --accounts acct_001,acct_002,acct_003

  # RAG extraction with reviewer disabled
  python -m core.scalable_pipeline --accounts acct_001 --extraction-backend rag --no-reviewer
        """,
    )
    parser.add_argument("--accounts", required=True, help="Comma-separated account IDs")
    parser.add_argument("--pdf-dir", default=None, help="PDF directory (single account)")
    parser.add_argument("--phase", default="all", choices=["all", "phase0", "phase1", "phase2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--discovery-mode", default="rlm_v3_unified",
                        choices=["standard", "rlm_v3", "rlm_v3_unified", "hybrid"])
    parser.add_argument("--extraction-backend", default="rag", choices=["rag", "sequential"])
    parser.add_argument("--storage-backend", default="postgres", choices=["postgres", "duckdb"],
                        help="Storage backend: postgres (pgvector+BM25) or duckdb (file-per-account)")
    parser.add_argument("--pg-dsn", default=None,
                        help="PostgreSQL DSN (default: $PG_DSN or postgresql://localhost:5432/clustering_v2)")
    parser.add_argument("--max-llm-concurrency", type=int, default=8)
    parser.add_argument("--max-cluster-concurrency", type=int, default=4)
    parser.add_argument("--no-reviewer", action="store_true")
    parser.add_argument("--max-sample-docs", type=int, default=10)
    parser.add_argument("--rag-max-retries", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)-24s %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config = ScalableConfig(
        device=args.device,
        optuna_trials=args.optuna_trials,
        discovery_mode=args.discovery_mode,
        extraction_backend=args.extraction_backend,
        storage_backend=args.storage_backend,
        max_llm_concurrency=args.max_llm_concurrency,
        max_cluster_concurrency=args.max_cluster_concurrency,
        rag_enable_reviewer=not args.no_reviewer,
        max_sample_docs=args.max_sample_docs,
        rag_max_retry_rounds=args.rag_max_retries,
    )
    if args.pg_dsn:
        config.pg_dsn = args.pg_dsn

    account_ids = [a.strip() for a in args.accounts.split(",") if a.strip()]

    # Build pdf_dirs map (single dir → first account, or per-account convention)
    pdf_dirs = None
    if args.pdf_dir:
        if len(account_ids) == 1:
            pdf_dirs = {account_ids[0]: args.pdf_dir}
        else:
            # Convention: pdf_dir/<account_id>/ contains that account's PDFs
            pdf_dirs = {}
            for aid in account_ids:
                acct_dir = Path(args.pdf_dir) / aid
                if acct_dir.exists():
                    pdf_dirs[aid] = str(acct_dir)

    phases = [args.phase] if args.phase != "all" else None

    async def _run():
        pipeline = ScalablePipeline(config)
        try:
            result = await pipeline.run(
                account_ids=account_ids,
                pdf_dirs=pdf_dirs,
                phases=phases,
            )
            print(json.dumps(result, indent=2, default=str))
            return 0 if all(
                "error" not in v for v in result.get("accounts", {}).values()
            ) else 1
        finally:
            await pipeline.close()

    return asyncio.run(_run())


if __name__ == "__main__":
    import sys
    sys.exit(main())
