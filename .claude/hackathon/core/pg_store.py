"""
PostgreSQL + pgvector + pg_bm25 storage layer — replaces DuckDB + LanceDB.
==========================================================================

Designed for 1M docs / 10K accounts at scale. Replaces:
  - DuckDB per-account files  → PG schema-per-account (shared cluster)
  - LanceDB per-account dirs  → pgvector HNSW indexes (pre-filtered ANN)
  - Shared checkpoint DuckDB  → PG checkpoints table (MVCC, multi-writer)
  - In-memory FAISS+BM25      → pg_bm25 / tsvector (persistent hybrid search)

Required PostgreSQL extensions (CREATE EXTENSION IF NOT EXISTS ...):
  - vector          (pgvector 0.7+)  — HNSW/IVFFlat ANN on embeddings
  - pg_trgm         (built-in)       — trigram similarity for fuzzy text match
  - btree_gin       (built-in)       — GIN index support for BM25-style search

Optional (ParadeDB — install separately if available):
  - pg_search       (ParadeDB)       — native BM25 scoring via bm25 index

Connection model:
  - asyncpg pool (min 5, max 50) with pgbouncer in front for 10K accounts
  - schema-per-account: CREATE SCHEMA IF NOT EXISTS acct_{account_id}
  - shared schemas: public.checkpoints, public.pipeline_runs

Usage:
    from core.pg_store import PgStoreManager, PgConfig

    mgr = PgStoreManager(PgConfig(dsn="postgresql://..."))
    await mgr.initialize()

    store = await mgr.get_store("acct_001")
    await store.insert_agreement(agreement_id="agr_1", ...)
    results = await store.vector_search(query_vec, top_k=20, account_id="acct_001")
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PgConfig:
    """PostgreSQL connection + behavior configuration."""

    # Connection
    dsn: str = field(default_factory=lambda: os.getenv(
        "PG_DSN", "postgresql://localhost:5432/clustering_v2"
    ))
    min_pool_size: int = 5
    max_pool_size: int = 50
    statement_cache_size: int = 100

    # Schema strategy
    schema_strategy: str = "per_account"   # per_account | shared
    shared_schema: str = "public"

    # Vector config
    embedding_dim: int = 768
    hnsw_m: int = 16                       # HNSW graph connectivity
    hnsw_ef_construction: int = 128        # HNSW build-time search width
    hnsw_ef_search: int = 100              # HNSW query-time search width
    distance_metric: str = "cosine"        # cosine | l2 | inner_product

    # BM25 / text search
    use_pg_search: bool = False            # True = ParadeDB pg_search, False = tsvector
    text_search_config: str = "english"

    # Connection timeouts
    connect_timeout: float = 10.0
    command_timeout: float = 60.0

    # Checkpoint sharding (no longer needed — PG is multi-writer)
    checkpoint_table: str = "checkpoints"


def _generate_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def _sanitize_schema(account_id: str) -> str:
    """Sanitize account_id into a valid PG schema name."""
    # Only allow alphanumeric + underscore, prefix with acct_
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in account_id)
    return f"acct_{sanitized}"


# ══════════════════════════════════════════════════════════════════════════════
# §2  PG STORE MANAGER — Connection pool + schema lifecycle
# ══════════════════════════════════════════════════════════════════════════════


class PgStoreManager:
    """
    Top-level manager: owns the asyncpg pool, provisions schemas,
    and hands out per-account PgStore instances.

    Lifecycle:
      mgr = PgStoreManager(config)
      await mgr.initialize()          # create pool + extensions + shared tables
      store = await mgr.get_store("acct_001")  # provisions schema if needed
      ...
      await mgr.close()
    """

    def __init__(self, config: PgConfig | None = None):
        self.config = config or PgConfig()
        self._pool = None
        self._initialized_schemas: set[str] = set()
        self._schema_lock = asyncio.Lock()

    async def initialize(self):
        """Create connection pool, install extensions, create shared tables."""
        import asyncpg

        self._pool = await asyncpg.create_pool(
            dsn=self.config.dsn,
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
            command_timeout=self.config.command_timeout,
            statement_cache_size=self.config.statement_cache_size,
        )
        logger.info(
            f"PG pool created: {self.config.min_pool_size}-{self.config.max_pool_size} connections"
        )

        await self._install_extensions()
        await self._create_shared_tables()

    async def _install_extensions(self):
        """Install required PG extensions (idempotent)."""
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS btree_gin")

            if self.config.use_pg_search:
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
                    logger.info("ParadeDB pg_search extension installed")
                except Exception as e:
                    logger.warning(
                        f"pg_search not available, falling back to tsvector: {e}"
                    )
                    self.config.use_pg_search = False

    async def _create_shared_tables(self):
        """Create cross-account tables in public schema."""
        async with self._pool.acquire() as conn:
            # Checkpoint table (replaces shared DuckDB bottleneck)
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.shared_schema}.{self.config.checkpoint_table} (
                    checkpoint_key  TEXT PRIMARY KEY,
                    phase           TEXT NOT NULL,
                    account_id      TEXT NOT NULL,
                    run_id          TEXT NOT NULL,
                    unit_id         TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    artifact_path   TEXT,
                    metadata        JSONB DEFAULT '{{}}'::jsonb,
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    error           TEXT
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_cp_account_phase
                ON {self.config.shared_schema}.{self.config.checkpoint_table}
                (account_id, phase, status)
            """)

            # Pipeline runs
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.shared_schema}.pipeline_runs (
                    run_id      TEXT PRIMARY KEY,
                    account_id  TEXT NOT NULL,
                    status      TEXT DEFAULT 'running',
                    stage       TEXT,
                    started_at  TIMESTAMPTZ DEFAULT NOW(),
                    ended_at    TIMESTAMPTZ,
                    metadata    JSONB DEFAULT '{{}}'::jsonb
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_account
                ON pipeline_runs (account_id, status)
            """)

    async def get_store(self, account_id: str) -> "PgStore":
        """Get or create an account-scoped store. Provisions schema on first access."""
        schema = _sanitize_schema(account_id)

        if schema not in self._initialized_schemas:
            async with self._schema_lock:
                if schema not in self._initialized_schemas:
                    await self._provision_schema(schema)
                    self._initialized_schemas.add(schema)

        return PgStore(
            pool=self._pool,
            schema=schema,
            account_id=account_id,
            config=self.config,
        )

    async def get_checkpoint_manager(self) -> "PgCheckpointManager":
        """Get the shared checkpoint manager."""
        return PgCheckpointManager(pool=self._pool, config=self.config)

    async def get_feedback_store(self, tenant_id: str) -> "PgFeedbackStore":
        """Get a tenant-scoped feedback store."""
        schema = _sanitize_schema(tenant_id)
        if schema not in self._initialized_schemas:
            async with self._schema_lock:
                if schema not in self._initialized_schemas:
                    await self._provision_schema(schema)
                    self._initialized_schemas.add(schema)
        return PgFeedbackStore(pool=self._pool, schema=schema, tenant_id=tenant_id)

    async def _provision_schema(self, schema: str):
        """Create schema + all tables + indexes + vector indexes for one account."""
        dim = self.config.embedding_dim
        dist_fn = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "inner_product": "vector_ip_ops",
        }.get(self.config.distance_metric, "vector_cosine_ops")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

                # ── Core tables ──

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.domains (
                        domain_id       TEXT PRIMARY KEY,
                        label           TEXT NOT NULL,
                        description     TEXT DEFAULT '',
                        agreement_count INTEGER DEFAULT 0,
                        confidence      DOUBLE PRECISION DEFAULT 0.0,
                        centroid        vector({dim}),
                        created_at      TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.agreements (
                        agreement_id      TEXT PRIMARY KEY,
                        filename          TEXT NOT NULL,
                        domain_id         TEXT,
                        raw_text          TEXT,
                        summary_text      TEXT DEFAULT '',
                        domain_confidence DOUBLE PRECISION,
                        created_at        TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_agreements_domain
                    ON {schema}.agreements (domain_id)
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.chunks (
                        chunk_id      TEXT PRIMARY KEY,
                        agreement_id  TEXT NOT NULL,
                        clause_id     TEXT,
                        chunk_text    TEXT NOT NULL,
                        chunk_index   INTEGER DEFAULT 0,
                        embedding     vector({dim}),
                        token_count   INTEGER DEFAULT 0,
                        embedding_dim INTEGER DEFAULT {dim},
                        tsv           tsvector GENERATED ALWAYS AS (
                            to_tsvector('{self.config.text_search_config}', chunk_text)
                        ) STORED
                    )
                """)
                # B-tree indexes for foreign-key lookups
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_chunks_agreement
                    ON {schema}.chunks (agreement_id)
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_chunks_clause
                    ON {schema}.chunks (agreement_id, clause_id)
                """)
                # GIN index on tsvector for full-text BM25-style search
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_chunks_tsv
                    ON {schema}.chunks USING GIN (tsv)
                """)
                # HNSW vector index for ANN (pre-filtered by agreement_id)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                    ON {schema}.chunks USING hnsw (embedding {dist_fn})
                    WITH (m = {self.config.hnsw_m}, ef_construction = {self.config.hnsw_ef_construction})
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.clauses (
                        clause_id       TEXT PRIMARY KEY,
                        agreement_id    TEXT NOT NULL,
                        clause_number   TEXT DEFAULT '',
                        clause_title    TEXT DEFAULT '',
                        heading_level   INTEGER DEFAULT 0,
                        clause_index    INTEGER DEFAULT 0,
                        clause_type_id  TEXT,
                        full_text       TEXT DEFAULT '',
                        token_count     INTEGER DEFAULT 0
                    )
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_clauses_agreement
                    ON {schema}.clauses (agreement_id)
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.clusters (
                        cluster_id      TEXT PRIMARY KEY,
                        domain_id       TEXT,
                        label           TEXT DEFAULT '',
                        description     TEXT DEFAULT '',
                        keywords        JSONB DEFAULT '[]'::jsonb,
                        aliases         JSONB DEFAULT '[]'::jsonb,
                        chunk_count     INTEGER DEFAULT 0,
                        agreement_count INTEGER DEFAULT 0,
                        quality_score   DOUBLE PRECISION,
                        centroid        vector({dim}),
                        merged_from     JSONB DEFAULT '[]'::jsonb,
                        created_at      TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_clusters_domain
                    ON {schema}.clusters (domain_id)
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.cluster_assignments (
                        chunk_id    TEXT NOT NULL,
                        cluster_id  TEXT NOT NULL,
                        probability DOUBLE PRECISION DEFAULT 1.0,
                        is_outlier  BOOLEAN DEFAULT FALSE,
                        PRIMARY KEY (chunk_id, cluster_id)
                    )
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_ca_cluster
                    ON {schema}.cluster_assignments (cluster_id)
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_ca_outlier
                    ON {schema}.cluster_assignments (is_outlier)
                    WHERE is_outlier = TRUE
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.field_definitions (
                        field_id    TEXT PRIMARY KEY,
                        cluster_id  TEXT,
                        name        TEXT NOT NULL,
                        field_type  TEXT NOT NULL,
                        description TEXT DEFAULT '',
                        source      TEXT DEFAULT 'auto_discovered',
                        examples    JSONB DEFAULT '[]'::jsonb,
                        created_at  TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_fields_cluster
                    ON {schema}.field_definitions (cluster_id)
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.extractions (
                        extraction_id   TEXT PRIMARY KEY,
                        agreement_id    TEXT NOT NULL,
                        field_id        TEXT NOT NULL,
                        value           TEXT,
                        confidence      DOUBLE PRECISION DEFAULT 0.0,
                        source_chunk_id TEXT,
                        was_corrected   BOOLEAN DEFAULT FALSE,
                        original_value  TEXT,
                        extracted_at    TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_extractions_agreement
                    ON {schema}.extractions (agreement_id)
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_extractions_field
                    ON {schema}.extractions (field_id)
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_extractions_agreement_field
                    ON {schema}.extractions (agreement_id, field_id)
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.composite_definitions (
                        field_id         TEXT PRIMARY KEY,
                        name             TEXT NOT NULL,
                        expression       TEXT NOT NULL,
                        source_field_ids JSONB NOT NULL,
                        result_type      TEXT DEFAULT 'float',
                        cluster_id       TEXT,
                        created_at       TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.corrections (
                        correction_id TEXT PRIMARY KEY,
                        extraction_id TEXT,
                        field_id      TEXT,
                        old_value     TEXT,
                        new_value     TEXT,
                        corrected_at  TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.cluster_centroids (
                        cluster_id   TEXT PRIMARY KEY,
                        centroid     vector({dim}),
                        chunk_count  INTEGER DEFAULT 0,
                        last_updated TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.novel_documents (
                        doc_id     TEXT PRIMARY KEY,
                        raw_text   TEXT,
                        embedding  vector({dim}),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Intent tables
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.intent_types (
                        intent_type_id   TEXT PRIMARY KEY,
                        label            TEXT NOT NULL,
                        description      TEXT DEFAULT '',
                        sample_attributes JSONB DEFAULT '[]'::jsonb,
                        aliases          JSONB DEFAULT '[]'::jsonb,
                        occurrence_count INTEGER DEFAULT 1,
                        created_at       TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.clause_intents (
                        intent_id        TEXT PRIMARY KEY,
                        clause_id        TEXT NOT NULL,
                        agreement_id     TEXT NOT NULL,
                        clause_type_id   TEXT,
                        intent_type_id   TEXT,
                        intent_label     TEXT,
                        summary          TEXT DEFAULT '',
                        party_from       TEXT DEFAULT 'none',
                        party_to         TEXT DEFAULT 'none',
                        attributes       JSONB DEFAULT '{{}}'::jsonb,
                        intent_hash      TEXT DEFAULT '',
                        created_at       TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_ci_agreement
                    ON {schema}.clause_intents (agreement_id)
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_ci_clause
                    ON {schema}.clause_intents (clause_id)
                """)

                # EVoC cluster layers
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.cluster_layers (
                        clustering_run TEXT NOT NULL,
                        layer_index    INTEGER NOT NULL,
                        item_index     INTEGER NOT NULL,
                        item_id        TEXT,
                        cluster_label  INTEGER NOT NULL,
                        PRIMARY KEY (clustering_run, layer_index, item_index)
                    )
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.cluster_layer_meta (
                        clustering_run    TEXT NOT NULL,
                        layer_index       INTEGER NOT NULL,
                        n_clusters        INTEGER,
                        n_outliers        INTEGER,
                        persistence_score DOUBLE PRECISION,
                        silhouette_score  DOUBLE PRECISION,
                        cosine_score      DOUBLE PRECISION,
                        composite_score   DOUBLE PRECISION,
                        is_selected       BOOLEAN DEFAULT FALSE,
                        PRIMARY KEY (clustering_run, layer_index)
                    )
                """)

                # Document vectors (for doc-level ANN search)
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.documents (
                        agreement_id TEXT PRIMARY KEY,
                        filename     TEXT DEFAULT '',
                        summary      TEXT DEFAULT '',
                        vector       vector({dim})
                    )
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_documents_vector
                    ON {schema}.documents USING hnsw (vector {dist_fn})
                    WITH (m = {self.config.hnsw_m}, ef_construction = {self.config.hnsw_ef_construction})
                """)

                # Feedback tables (in same schema — no separate sidecar DB needed)
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.feedback_events (
                        event_id      TEXT PRIMARY KEY,
                        tenant_id     TEXT NOT NULL,
                        user_id       TEXT NOT NULL,
                        entity_type   TEXT NOT NULL,
                        entity_id     TEXT NOT NULL,
                        action        TEXT NOT NULL,
                        context       JSONB DEFAULT '{{}}'::jsonb,
                        old_value     JSONB DEFAULT '{{}}'::jsonb,
                        new_value     JSONB DEFAULT '{{}}'::jsonb,
                        source_doc_id TEXT,
                        created_at    TIMESTAMPTZ DEFAULT NOW(),
                        session_id    TEXT
                    )
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_fb_entity
                    ON {schema}.feedback_events (entity_type, entity_id)
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.personalization_weights (
                        weight_id      TEXT PRIMARY KEY,
                        tenant_id      TEXT NOT NULL,
                        entity_type    TEXT NOT NULL,
                        entity_id      TEXT NOT NULL,
                        weight         DOUBLE PRECISION DEFAULT 1.0,
                        adjustment     DOUBLE PRECISION DEFAULT 0.0,
                        feedback_count INTEGER DEFAULT 0,
                        last_updated   TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.virtual_clusters (
                        cluster_id    TEXT PRIMARY KEY,
                        tenant_id     TEXT NOT NULL,
                        label         TEXT NOT NULL,
                        description   TEXT DEFAULT '',
                        centroid      vector({dim}),
                        source_action TEXT DEFAULT 'user_created',
                        member_count  INTEGER DEFAULT 0,
                        created_by    TEXT,
                        created_at    TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.virtual_assignments (
                        assignment_id      TEXT PRIMARY KEY,
                        tenant_id          TEXT NOT NULL,
                        chunk_id           TEXT,
                        clause_id          TEXT,
                        global_cluster_id  TEXT,
                        virtual_cluster_id TEXT NOT NULL,
                        similarity         DOUBLE PRECISION,
                        assigned_by        TEXT,
                        assigned_at        TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.intent_overrides (
                        override_id             TEXT PRIMARY KEY,
                        tenant_id               TEXT NOT NULL,
                        global_intent_label     TEXT NOT NULL,
                        tenant_intent_label     TEXT NOT NULL,
                        tenant_summary          TEXT DEFAULT '',
                        apply_to_clause_types   JSONB DEFAULT '[]'::jsonb,
                        created_by              TEXT,
                        created_at              TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.tenant_config (
                        tenant_id                   TEXT PRIMARY KEY,
                        high_confidence_threshold   DOUBLE PRECISION DEFAULT 0.85,
                        tentative_threshold         DOUBLE PRECISION DEFAULT 0.60,
                        outlier_min_similarity      DOUBLE PRECISION DEFAULT 0.35,
                        intent_match_threshold      DOUBLE PRECISION DEFAULT 0.85,
                        intent_temperature          DOUBLE PRECISION DEFAULT 0.0,
                        edge_decay_half_life_days   INTEGER DEFAULT 90,
                        auto_apply_corrections      BOOLEAN DEFAULT FALSE,
                        updated_at                  TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

        logger.info(f"Schema provisioned: {schema}")

    async def drop_schema(self, account_id: str):
        """Drop an account schema entirely. USE WITH CAUTION."""
        schema = _sanitize_schema(account_id)
        async with self._pool.acquire() as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        self._initialized_schemas.discard(schema)
        logger.info(f"Schema dropped: {schema}")

    async def list_schemas(self) -> list[str]:
        """List all account schemas."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name LIKE 'acct_%' ORDER BY schema_name"
            )
        return [r["schema_name"] for r in rows]

    async def close(self):
        if self._pool:
            await self._pool.close()
            logger.info("PG pool closed")


# ══════════════════════════════════════════════════════════════════════════════
# §3  PG STORE — Account-scoped CRUD (drop-in for ClusteringStore)
# ══════════════════════════════════════════════════════════════════════════════


class PgStore:
    """
    Async account-scoped store backed by PostgreSQL + pgvector.

    API-compatible with ClusteringStore — same method names, async versions.
    All queries are scoped to the account's schema.
    """

    def __init__(self, pool, schema: str, account_id: str, config: PgConfig):
        self._pool = pool
        self._schema = schema
        self._account_id = account_id
        self._config = config
        self._dim = config.embedding_dim

    # ── Helpers ────────────────────────────────────────────────────────────

    def _table(self, name: str) -> str:
        return f"{self._schema}.{name}"

    @staticmethod
    def _vec_to_pg(arr: np.ndarray) -> str:
        """Convert numpy vector to pgvector literal string."""
        return "[" + ",".join(f"{v:.6f}" for v in arr.astype(np.float32).flat) + "]"

    @staticmethod
    def _pg_to_vec(pg_str: str) -> np.ndarray:
        """Convert pgvector string back to numpy array."""
        if pg_str is None:
            return np.array([])
        if isinstance(pg_str, (list, np.ndarray)):
            return np.array(pg_str, dtype=np.float32)
        cleaned = pg_str.strip("[]")
        return np.array([float(x) for x in cleaned.split(",")], dtype=np.float32)

    @staticmethod
    def _row_to_dict(record) -> dict:
        """Convert asyncpg Record to dict."""
        return dict(record)

    # ── Domain CRUD ────────────────────────────────────────────────────────

    async def upsert_domain(
        self, domain_id: str, label: str, description: str = "",
        agreement_count: int = 0, confidence: float = 0.0,
        centroid: np.ndarray | None = None,
    ):
        centroid_str = self._vec_to_pg(centroid) if centroid is not None else None
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('domains')}
                (domain_id, label, description, agreement_count, confidence, centroid)
                VALUES ($1, $2, $3, $4, $5, $6::vector)
                ON CONFLICT (domain_id) DO UPDATE SET
                    label = EXCLUDED.label,
                    description = EXCLUDED.description,
                    agreement_count = EXCLUDED.agreement_count,
                    confidence = EXCLUDED.confidence,
                    centroid = EXCLUDED.centroid
            """, domain_id, label, description, agreement_count, confidence, centroid_str)

    async def get_domains(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self._table('domains')} ORDER BY agreement_count DESC"
            )
        return [self._row_to_dict(r) for r in rows]

    # ── Agreement CRUD ─────────────────────────────────────────────────────

    async def insert_agreement(
        self, agreement_id: str, filename: str, raw_text: str,
        summary_text: str = "", domain_id: str | None = None,
        domain_confidence: float | None = None,
    ):
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('agreements')}
                (agreement_id, filename, domain_id, raw_text, summary_text, domain_confidence)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (agreement_id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    domain_id = EXCLUDED.domain_id,
                    raw_text = EXCLUDED.raw_text,
                    summary_text = EXCLUDED.summary_text,
                    domain_confidence = EXCLUDED.domain_confidence
            """, agreement_id, filename, domain_id, raw_text, summary_text, domain_confidence)

    async def get_agreements(self, domain_id: str | None = None) -> list[dict]:
        async with self._pool.acquire() as conn:
            if domain_id:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('agreements')} WHERE domain_id = $1 ORDER BY filename",
                    domain_id,
                )
            else:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('agreements')} ORDER BY filename"
                )
        return [self._row_to_dict(r) for r in rows]

    async def get_agreement(self, agreement_id: str) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._table('agreements')} WHERE agreement_id = $1",
                agreement_id,
            )
        return self._row_to_dict(row) if row else None

    async def update_agreement_domain(self, agreement_id: str, domain_id: str, confidence: float):
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {self._table('agreements')} SET domain_id = $1, domain_confidence = $2 WHERE agreement_id = $3",
                domain_id, confidence, agreement_id,
            )

    # ── Chunk CRUD ─────────────────────────────────────────────────────────

    async def insert_chunk(
        self, chunk_id: str, agreement_id: str, chunk_text: str,
        chunk_index: int, embedding: np.ndarray | None = None,
        token_count: int = 0, clause_id: str | None = None,
    ):
        emb_str = self._vec_to_pg(embedding) if embedding is not None else None
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('chunks')}
                (chunk_id, agreement_id, clause_id, chunk_text, chunk_index, embedding, token_count, embedding_dim)
                VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    embedding = EXCLUDED.embedding,
                    token_count = EXCLUDED.token_count
            """, chunk_id, agreement_id, clause_id, chunk_text, chunk_index,
                emb_str, token_count, self._dim)

    async def insert_chunks_batch(
        self,
        chunk_ids: list[str],
        agreement_ids: list[str],
        texts: list[str],
        chunk_indexes: list[int],
        embeddings: np.ndarray | None = None,
        token_counts: list[int] | None = None,
        clause_ids: list[str | None] | None = None,
    ):
        """Batch insert chunks using COPY for maximum throughput."""
        n = len(chunk_ids)
        _token_counts = token_counts or [0] * n
        _clause_ids = clause_ids or [None] * n

        records = []
        for i in range(n):
            emb_str = self._vec_to_pg(embeddings[i]) if embeddings is not None else None
            records.append((
                chunk_ids[i], agreement_ids[i], _clause_ids[i],
                texts[i], chunk_indexes[i], emb_str,
                _token_counts[i], self._dim,
            ))

        async with self._pool.acquire() as conn:
            # Use executemany for batch upsert
            await conn.executemany(f"""
                INSERT INTO {self._table('chunks')}
                (chunk_id, agreement_id, clause_id, chunk_text, chunk_index, embedding, token_count, embedding_dim)
                VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    embedding = EXCLUDED.embedding,
                    token_count = EXCLUDED.token_count
            """, records)

        logger.debug(f"Batch inserted {n} chunks into {self._schema}")

    async def get_chunks(
        self, agreement_id: str | None = None, cluster_id: str | None = None,
        include_outliers: bool = True,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            if cluster_id:
                outlier_filter = "" if include_outliers else " AND (ca.is_outlier = FALSE OR ca.is_outlier IS NULL)"
                rows = await conn.fetch(f"""
                    SELECT c.*, ca.is_outlier FROM {self._table('chunks')} c
                    JOIN {self._table('cluster_assignments')} ca ON c.chunk_id = ca.chunk_id
                    WHERE ca.cluster_id = $1{outlier_filter}
                    ORDER BY c.chunk_index
                """, cluster_id)
            elif agreement_id:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('chunks')} WHERE agreement_id = $1 ORDER BY chunk_index",
                    agreement_id,
                )
            else:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('chunks')} ORDER BY agreement_id, chunk_index"
                )
        return [self._row_to_dict(r) for r in rows]

    async def get_all_chunk_embeddings(
        self, domain_id: str | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """Return (chunk_ids, embeddings_matrix) for all chunks with embeddings."""
        async with self._pool.acquire() as conn:
            if domain_id:
                rows = await conn.fetch(f"""
                    SELECT c.chunk_id, c.embedding::text FROM {self._table('chunks')} c
                    JOIN {self._table('agreements')} a ON c.agreement_id = a.agreement_id
                    WHERE a.domain_id = $1 AND c.embedding IS NOT NULL
                """, domain_id)
            else:
                rows = await conn.fetch(f"""
                    SELECT chunk_id, embedding::text FROM {self._table('chunks')}
                    WHERE embedding IS NOT NULL
                """)

        if not rows:
            return [], np.array([])

        chunk_ids = [r["chunk_id"] for r in rows]
        embeddings = np.array([self._pg_to_vec(r["embedding"]) for r in rows])
        return chunk_ids, embeddings

    # ── Vector Search (replaces LanceDB + in-memory FAISS) ─────────────────

    async def vector_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        agreement_id: str | None = None,
        domain_id: str | None = None,
        cluster_id: str | None = None,
    ) -> list[dict]:
        """
        pgvector HNSW search with pre-filtering.

        Unlike LanceDB post-filter, pgvector applies WHERE *before* the ANN
        scan when you have matching partial indexes or the planner decides
        a filtered scan is cheaper.
        """
        vec_str = self._vec_to_pg(query_vector)
        dist_op = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}.get(
            self._config.distance_metric, "<=>"
        )

        # Build WHERE clause for pre-filtering
        conditions = []
        params: list[Any] = [vec_str]
        param_idx = 2

        if agreement_id:
            conditions.append(f"c.agreement_id = ${param_idx}")
            params.append(agreement_id)
            param_idx += 1
        if domain_id:
            conditions.append(f"""c.agreement_id IN (
                SELECT agreement_id FROM {self._table('agreements')} WHERE domain_id = ${param_idx}
            )""")
            params.append(domain_id)
            param_idx += 1
        if cluster_id:
            conditions.append(f"""c.chunk_id IN (
                SELECT chunk_id FROM {self._table('cluster_assignments')} WHERE cluster_id = ${param_idx}
            )""")
            params.append(cluster_id)
            param_idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        async with self._pool.acquire() as conn:
            # Set HNSW search parameters for this query
            await conn.execute(
                f"SET LOCAL hnsw.ef_search = {self._config.hnsw_ef_search}"
            )
            rows = await conn.fetch(f"""
                SELECT c.chunk_id, c.agreement_id, c.chunk_text, c.chunk_index,
                       c.embedding {dist_op} $1::vector AS distance
                FROM {self._table('chunks')} c
                {where}
                ORDER BY c.embedding {dist_op} $1::vector
                LIMIT {top_k}
            """, *params)

        return [
            {
                "chunk_id": r["chunk_id"],
                "agreement_id": r["agreement_id"],
                "text": r["chunk_text"],
                "chunk_index": r["chunk_index"],
                "score": float(r["distance"]),
            }
            for r in rows
        ]

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 20,
        agreement_id: str | None = None,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Hybrid search: pgvector ANN + tsvector BM25 + RRF fusion.

        Replaces the in-memory HybridRetriever (FAISS + BM25Okapi) with a
        persistent, pre-indexed search that doesn't need to be rebuilt.
        """
        vec_str = self._vec_to_pg(query_vector)
        dist_op = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}.get(
            self._config.distance_metric, "<=>"
        )

        # Build agreement filter
        agreement_filter = ""
        params_sem: list[Any] = [vec_str]
        params_bm25: list[Any] = [query_text]

        if agreement_id:
            agreement_filter = "AND agreement_id = $2"
            params_sem.append(agreement_id)
            params_bm25.append(agreement_id)

        fetch_k = top_k * 3  # over-fetch for RRF

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"SET LOCAL hnsw.ef_search = {self._config.hnsw_ef_search}"
            )

            # Semantic search
            sem_rows = await conn.fetch(f"""
                SELECT chunk_id, chunk_text, agreement_id,
                       embedding {dist_op} $1::vector AS distance
                FROM {self._table('chunks')}
                WHERE embedding IS NOT NULL {agreement_filter}
                ORDER BY embedding {dist_op} $1::vector
                LIMIT {fetch_k}
            """, *params_sem)

            # BM25 / tsvector search
            bm25_rows = await conn.fetch(f"""
                SELECT chunk_id, chunk_text, agreement_id,
                       ts_rank_cd(tsv, plainto_tsquery('{self._config.text_search_config}', $1)) AS rank
                FROM {self._table('chunks')}
                WHERE tsv @@ plainto_tsquery('{self._config.text_search_config}', $1) {agreement_filter}
                ORDER BY rank DESC
                LIMIT {fetch_k}
            """, *params_bm25)

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        for rank, r in enumerate(sem_rows):
            cid = r["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + semantic_weight / (rrf_k + rank)
            chunk_data[cid] = {
                "chunk_id": cid,
                "text": r["chunk_text"],
                "agreement_id": r["agreement_id"],
            }

        for rank, r in enumerate(bm25_rows):
            cid = r["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + bm25_weight / (rrf_k + rank)
            if cid not in chunk_data:
                chunk_data[cid] = {
                    "chunk_id": cid,
                    "text": r["chunk_text"],
                    "agreement_id": r["agreement_id"],
                }

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [
            {**chunk_data[cid], "score": rrf_scores[cid], "source": "hybrid_rrf"}
            for cid in sorted_ids
        ]

    # ── Document vectors ───────────────────────────────────────────────────

    async def upsert_documents(
        self,
        agreement_ids: list[str],
        vectors: np.ndarray,
        filenames: list[str],
        summaries: list[str],
    ) -> int:
        records = [
            (agreement_ids[i], filenames[i], summaries[i], self._vec_to_pg(vectors[i]))
            for i in range(len(agreement_ids))
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(f"""
                INSERT INTO {self._table('documents')}
                (agreement_id, filename, summary, vector)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (agreement_id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    summary = EXCLUDED.summary,
                    vector = EXCLUDED.vector
            """, records)
        return len(records)

    async def search_documents(
        self, query_vector: np.ndarray, top_k: int = 10,
    ) -> list[dict]:
        vec_str = self._vec_to_pg(query_vector)
        dist_op = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}.get(
            self._config.distance_metric, "<=>"
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT agreement_id, filename, summary,
                       vector {dist_op} $1::vector AS distance
                FROM {self._table('documents')}
                ORDER BY vector {dist_op} $1::vector
                LIMIT {top_k}
            """, vec_str)
        return [
            {
                "agreement_id": r["agreement_id"],
                "filename": r.get("filename", ""),
                "summary": r.get("summary", ""),
                "score": float(r["distance"]),
            }
            for r in rows
        ]

    # ── Clause CRUD ────────────────────────────────────────────────────────

    async def insert_clause(
        self, clause_id: str, agreement_id: str,
        clause_number: str = "", clause_title: str = "",
        heading_level: int = 0, clause_index: int = 0,
        full_text: str = "", token_count: int = 0,
    ):
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('clauses')}
                (clause_id, agreement_id, clause_number, clause_title,
                 heading_level, clause_index, full_text, token_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (clause_id) DO UPDATE SET
                    clause_number = EXCLUDED.clause_number,
                    clause_title = EXCLUDED.clause_title,
                    full_text = EXCLUDED.full_text
            """, clause_id, agreement_id, clause_number, clause_title,
                heading_level, clause_index, full_text, token_count)

    async def get_clauses(self, agreement_id: str | None = None) -> list[dict]:
        async with self._pool.acquire() as conn:
            if agreement_id:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('clauses')} WHERE agreement_id = $1 ORDER BY clause_index",
                    agreement_id,
                )
            else:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('clauses')} ORDER BY agreement_id, clause_index"
                )
        return [self._row_to_dict(r) for r in rows]

    # ── Cluster CRUD ───────────────────────────────────────────────────────

    async def upsert_cluster(
        self, cluster_id: str, domain_id: str, label: str = "",
        description: str = "", keywords: list[str] | None = None,
        chunk_count: int = 0, agreement_count: int = 0,
        quality_score: float | None = None, centroid: np.ndarray | None = None,
        aliases: list[str] | None = None, merged_from: list[str] | None = None,
    ):
        centroid_str = self._vec_to_pg(centroid) if centroid is not None else None
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('clusters')}
                (cluster_id, domain_id, label, description, keywords, chunk_count,
                 agreement_count, quality_score, centroid, aliases, merged_from)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9::vector, $10::jsonb, $11::jsonb)
                ON CONFLICT (cluster_id) DO UPDATE SET
                    label = EXCLUDED.label,
                    description = EXCLUDED.description,
                    keywords = EXCLUDED.keywords,
                    chunk_count = EXCLUDED.chunk_count,
                    agreement_count = EXCLUDED.agreement_count,
                    quality_score = EXCLUDED.quality_score,
                    centroid = EXCLUDED.centroid,
                    aliases = EXCLUDED.aliases,
                    merged_from = EXCLUDED.merged_from
            """, cluster_id, domain_id, label, description,
                json.dumps(keywords or []), chunk_count, agreement_count,
                quality_score, centroid_str,
                json.dumps(aliases or []), json.dumps(merged_from or []))

    async def get_clusters(self, domain_id: str | None = None) -> list[dict]:
        async with self._pool.acquire() as conn:
            if domain_id:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('clusters')} WHERE domain_id = $1 ORDER BY chunk_count DESC",
                    domain_id,
                )
            else:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('clusters')} ORDER BY domain_id, chunk_count DESC"
                )
        results = []
        for r in rows:
            d = self._row_to_dict(r)
            # JSONB columns are already parsed by asyncpg
            if isinstance(d.get("keywords"), str):
                d["keywords"] = json.loads(d["keywords"])
            if isinstance(d.get("aliases"), str):
                d["aliases"] = json.loads(d["aliases"])
            if isinstance(d.get("merged_from"), str):
                d["merged_from"] = json.loads(d["merged_from"])
            results.append(d)
        return results

    async def get_cluster(self, cluster_id: str) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._table('clusters')} WHERE cluster_id = $1",
                cluster_id,
            )
        if not row:
            return None
        d = self._row_to_dict(row)
        if isinstance(d.get("keywords"), str):
            d["keywords"] = json.loads(d["keywords"])
        if isinstance(d.get("aliases"), str):
            d["aliases"] = json.loads(d["aliases"])
        if isinstance(d.get("merged_from"), str):
            d["merged_from"] = json.loads(d["merged_from"])
        return d

    # ── Cluster Assignments ────────────────────────────────────────────────

    async def insert_cluster_assignment(
        self, chunk_id: str, cluster_id: str,
        probability: float = 1.0, is_outlier: bool = False,
    ):
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('cluster_assignments')}
                (chunk_id, cluster_id, probability, is_outlier)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (chunk_id, cluster_id) DO UPDATE SET
                    probability = EXCLUDED.probability,
                    is_outlier = EXCLUDED.is_outlier
            """, chunk_id, cluster_id, probability, is_outlier)

    async def insert_cluster_assignments_batch(
        self,
        chunk_ids: list[str],
        cluster_ids: list[str],
        probabilities: list[float] | None = None,
        is_outliers: list[bool] | None = None,
    ):
        """Batch insert cluster assignments."""
        n = len(chunk_ids)
        _probs = probabilities or [1.0] * n
        _outliers = is_outliers or [False] * n
        records = [
            (chunk_ids[i], cluster_ids[i], _probs[i], _outliers[i])
            for i in range(n)
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(f"""
                INSERT INTO {self._table('cluster_assignments')}
                (chunk_id, cluster_id, probability, is_outlier)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (chunk_id, cluster_id) DO UPDATE SET
                    probability = EXCLUDED.probability,
                    is_outlier = EXCLUDED.is_outlier
            """, records)

    # ── Field Definitions ──────────────────────────────────────────────────

    async def upsert_field(
        self, field_id: str, cluster_id: str, name: str, field_type: str,
        description: str = "", source: str = "auto_discovered",
        examples: list | None = None,
    ):
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('field_definitions')}
                (field_id, cluster_id, name, field_type, description, source, examples)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                ON CONFLICT (field_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    field_type = EXCLUDED.field_type,
                    description = EXCLUDED.description,
                    examples = EXCLUDED.examples
            """, field_id, cluster_id, name, field_type, description, source,
                json.dumps(examples or []))

    async def get_fields(self, cluster_id: str | None = None) -> list[dict]:
        async with self._pool.acquire() as conn:
            if cluster_id:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('field_definitions')} WHERE cluster_id = $1 ORDER BY name",
                    cluster_id,
                )
            else:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('field_definitions')} ORDER BY cluster_id, name"
                )
        results = []
        for r in rows:
            d = self._row_to_dict(r)
            if isinstance(d.get("examples"), str):
                d["examples"] = json.loads(d["examples"])
            results.append(d)
        return results

    async def delete_field(self, field_id: str):
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._table('field_definitions')} WHERE field_id = $1",
                field_id,
            )

    # ── Extractions ────────────────────────────────────────────────────────

    async def insert_extraction(
        self, extraction_id: str, agreement_id: str, field_id: str,
        value: str | None, confidence: float = 0.0,
        source_chunk_id: str | None = None,
    ):
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('extractions')}
                (extraction_id, agreement_id, field_id, value, confidence, source_chunk_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (extraction_id) DO UPDATE SET
                    value = EXCLUDED.value,
                    confidence = EXCLUDED.confidence
            """, extraction_id, agreement_id, field_id, value, confidence, source_chunk_id)

    async def insert_extractions_batch(
        self,
        records: list[tuple[str, str, str, str | None, float, str | None]],
    ):
        """Batch insert extractions: (extraction_id, agreement_id, field_id, value, confidence, source_chunk_id)."""
        async with self._pool.acquire() as conn:
            await conn.executemany(f"""
                INSERT INTO {self._table('extractions')}
                (extraction_id, agreement_id, field_id, value, confidence, source_chunk_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (extraction_id) DO UPDATE SET
                    value = EXCLUDED.value,
                    confidence = EXCLUDED.confidence
            """, records)

    async def get_extractions(
        self, agreement_id: str | None = None,
        field_id: str | None = None,
        cluster_id: str | None = None,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            if cluster_id:
                rows = await conn.fetch(f"""
                    SELECT e.* FROM {self._table('extractions')} e
                    JOIN {self._table('field_definitions')} f ON e.field_id = f.field_id
                    WHERE f.cluster_id = $1
                    ORDER BY e.agreement_id
                """, cluster_id)
            else:
                conditions, params = [], []
                idx = 1
                if agreement_id:
                    conditions.append(f"agreement_id = ${idx}")
                    params.append(agreement_id)
                    idx += 1
                if field_id:
                    conditions.append(f"field_id = ${idx}")
                    params.append(field_id)
                    idx += 1
                where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('extractions')} {where} ORDER BY agreement_id",
                    *params,
                )
        return [self._row_to_dict(r) for r in rows]

    async def correct_extraction(self, extraction_id: str, new_value: str):
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT value, field_id FROM {self._table('extractions')} WHERE extraction_id = $1",
                extraction_id,
            )
            if not row:
                return
            old_value, field_id = row["value"], row["field_id"]
            await conn.execute(f"""
                UPDATE {self._table('extractions')}
                SET value = $1, was_corrected = TRUE, original_value = $2
                WHERE extraction_id = $3
            """, new_value, old_value, extraction_id)
            await conn.execute(f"""
                INSERT INTO {self._table('corrections')}
                (correction_id, extraction_id, field_id, old_value, new_value)
                VALUES ($1, $2, $3, $4, $5)
            """, _generate_id("corr_"), extraction_id, field_id, old_value, new_value)

    # ── Composite Definitions ──────────────────────────────────────────────

    async def upsert_composite(
        self, field_id: str, name: str, expression: str,
        source_field_ids: list[str], result_type: str = "float",
        cluster_id: str | None = None,
    ):
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('composite_definitions')}
                (field_id, name, expression, source_field_ids, result_type, cluster_id)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6)
                ON CONFLICT (field_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    expression = EXCLUDED.expression,
                    source_field_ids = EXCLUDED.source_field_ids
            """, field_id, name, expression, json.dumps(source_field_ids), result_type, cluster_id)

    async def get_composites(self, cluster_id: str | None = None) -> list[dict]:
        async with self._pool.acquire() as conn:
            if cluster_id:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('composite_definitions')} WHERE cluster_id = $1 ORDER BY name",
                    cluster_id,
                )
            else:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('composite_definitions')} ORDER BY name"
                )
        results = []
        for r in rows:
            d = self._row_to_dict(r)
            if isinstance(d.get("source_field_ids"), str):
                d["source_field_ids"] = json.loads(d["source_field_ids"])
            results.append(d)
        return results

    # ── Centroids ──────────────────────────────────────────────────────────

    async def save_centroid(self, cluster_id: str, centroid: np.ndarray, chunk_count: int):
        centroid_str = self._vec_to_pg(centroid)
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('cluster_centroids')}
                (cluster_id, centroid, chunk_count, last_updated)
                VALUES ($1, $2::vector, $3, NOW())
                ON CONFLICT (cluster_id) DO UPDATE SET
                    centroid = EXCLUDED.centroid,
                    chunk_count = EXCLUDED.chunk_count,
                    last_updated = NOW()
            """, cluster_id, centroid_str, chunk_count)

    async def get_all_centroids(self) -> dict[str, tuple[np.ndarray, int]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT cluster_id, centroid::text, chunk_count FROM {self._table('cluster_centroids')}"
            )
        return {
            r["cluster_id"]: (self._pg_to_vec(r["centroid"]), r["chunk_count"])
            for r in rows
        }

    # ── Novel Documents ────────────────────────────────────────────────────

    async def insert_novel_document(self, doc_id: str, raw_text: str, embedding: np.ndarray):
        emb_str = self._vec_to_pg(embedding)
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('novel_documents')}
                (doc_id, raw_text, embedding) VALUES ($1, $2, $3::vector)
                ON CONFLICT (doc_id) DO NOTHING
            """, doc_id, raw_text, emb_str)

    async def get_novel_documents(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self._table('novel_documents')} ORDER BY created_at"
            )
        return [self._row_to_dict(r) for r in rows]

    async def clear_novel_buffer(self):
        async with self._pool.acquire() as conn:
            await conn.execute(f"TRUNCATE {self._table('novel_documents')}")

    # ── Intent Types ───────────────────────────────────────────────────────

    async def upsert_intent_type(
        self, intent_type_id: str, label: str,
        description: str = "", sample_attributes: list[str] | None = None,
    ):
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('intent_types')}
                (intent_type_id, label, description, sample_attributes)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (intent_type_id) DO UPDATE SET
                    label = EXCLUDED.label,
                    description = EXCLUDED.description,
                    sample_attributes = EXCLUDED.sample_attributes
            """, intent_type_id, label, description,
                json.dumps(sample_attributes or []))

    async def get_intent_types(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self._table('intent_types')} ORDER BY occurrence_count DESC"
            )
        return [self._row_to_dict(r) for r in rows]

    # ── Clause Intents ─────────────────────────────────────────────────────

    async def insert_clause_intent(
        self, intent_id: str, clause_id: str,
        agreement_id: str, clause_type_id: str | None,
        intent_type_id: str | None, intent_label: str,
        summary: str = "", party_from: str = "none",
        party_to: str = "none", attributes: dict | None = None,
        intent_hash: str = "",
    ):
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('clause_intents')}
                (intent_id, clause_id, agreement_id, clause_type_id,
                 intent_type_id, intent_label, summary, party_from, party_to,
                 attributes, intent_hash)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11)
                ON CONFLICT (intent_id) DO UPDATE SET
                    intent_label = EXCLUDED.intent_label,
                    summary = EXCLUDED.summary,
                    attributes = EXCLUDED.attributes
            """, intent_id, clause_id, agreement_id, clause_type_id,
                intent_type_id, intent_label, summary, party_from, party_to,
                json.dumps(attributes or {}), intent_hash)

    async def get_clause_intents(
        self, clause_type_id: str | None = None,
        agreement_id: str | None = None,
        clause_id: str | None = None,
        intent_type_id: str | None = None,
    ) -> list[dict]:
        conditions, params = ["1=1"], []
        idx = 1
        if clause_type_id:
            conditions.append(f"clause_type_id = ${idx}")
            params.append(clause_type_id)
            idx += 1
        if agreement_id:
            conditions.append(f"agreement_id = ${idx}")
            params.append(agreement_id)
            idx += 1
        if clause_id:
            conditions.append(f"clause_id = ${idx}")
            params.append(clause_id)
            idx += 1
        if intent_type_id:
            conditions.append(f"intent_type_id = ${idx}")
            params.append(intent_type_id)
            idx += 1
        where = " AND ".join(conditions)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self._table('clause_intents')} WHERE {where} ORDER BY clause_id",
                *params,
            )
        return [self._row_to_dict(r) for r in rows]

    # ── Cluster Layers ─────────────────────────────────────────────────────

    async def save_cluster_layers(
        self, clustering_run: str, layers: list,
        item_ids: list[str], persistence_scores: list[float],
        selected_layer_idx: int, layer_scores: list[dict] | None = None,
    ):
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"DELETE FROM {self._table('cluster_layers')} WHERE clustering_run = $1",
                    clustering_run,
                )
                await conn.execute(
                    f"DELETE FROM {self._table('cluster_layer_meta')} WHERE clustering_run = $1",
                    clustering_run,
                )

                for li, layer in enumerate(layers):
                    labels = list(layer)
                    n_types = len(set(labels) - {-1})
                    n_out = sum(1 for t in labels if t == -1)
                    scores = layer_scores[li] if layer_scores and li < len(layer_scores) else {}

                    await conn.execute(f"""
                        INSERT INTO {self._table('cluster_layer_meta')}
                        (clustering_run, layer_index, n_clusters, n_outliers,
                         persistence_score, silhouette_score, cosine_score,
                         composite_score, is_selected)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """, clustering_run, li, n_types, n_out,
                        float(persistence_scores[li]) if li < len(persistence_scores) else None,
                        scores.get("silhouette"), scores.get("cosine"),
                        scores.get("composite"), li == selected_layer_idx)

                    # Batch insert layer assignments
                    records = [
                        (clustering_run, li, idx,
                         item_ids[idx] if idx < len(item_ids) else None,
                         int(labels[idx]))
                        for idx in range(len(labels))
                    ]
                    await conn.executemany(f"""
                        INSERT INTO {self._table('cluster_layers')}
                        (clustering_run, layer_index, item_index, item_id, cluster_label)
                        VALUES ($1, $2, $3, $4, $5)
                    """, records)

    async def get_cluster_layer_meta(self, clustering_run: str | None = None) -> list[dict]:
        async with self._pool.acquire() as conn:
            if clustering_run:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('cluster_layer_meta')} WHERE clustering_run = $1 ORDER BY layer_index",
                    clustering_run,
                )
            else:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._table('cluster_layer_meta')} ORDER BY clustering_run, layer_index"
                )
        return [self._row_to_dict(r) for r in rows]

    # ── Pipeline Runs ──────────────────────────────────────────────────────

    async def start_run(self, run_id: str, stage: str = "init"):
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO pipeline_runs (run_id, account_id, status, stage) VALUES ($1, $2, 'running', $3)",
                run_id, self._account_id, stage,
            )

    async def update_run(self, run_id: str, stage: str, status: str = "running"):
        async with self._pool.acquire() as conn:
            if status in ("completed", "failed"):
                await conn.execute(
                    "UPDATE pipeline_runs SET stage = $1, status = $2, ended_at = NOW() WHERE run_id = $3",
                    stage, status, run_id,
                )
            else:
                await conn.execute(
                    "UPDATE pipeline_runs SET stage = $1, status = $2 WHERE run_id = $3",
                    stage, status, run_id,
                )

    # ── Stats ──────────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        async with self._pool.acquire() as conn:
            counts = {}
            for table in ["domains", "agreements", "clusters", "field_definitions",
                          "extractions", "corrections", "composite_definitions"]:
                row = await conn.fetchval(f"SELECT COUNT(*) FROM {self._table(table)}")
                counts[table.replace("field_definitions", "fields").replace("composite_definitions", "composites")] = row

            avg_cov = await conn.fetchval(f"""
                SELECT AVG(coverage) FROM (
                    SELECT e.agreement_id,
                           COUNT(CASE WHEN e.value IS NOT NULL AND e.value != '' THEN 1 END)::float
                           / GREATEST(COUNT(*), 1) AS coverage
                    FROM {self._table('extractions')} e
                    GROUP BY e.agreement_id
                ) sub
            """)
        counts["avg_coverage"] = round(avg_cov, 3) if avg_cov else 0.0
        return counts

    def close(self):
        """No-op — pool is managed by PgStoreManager."""
        pass


# ══════════════════════════════════════════════════════════════════════════════
# §4  PG CHECKPOINT MANAGER — Multi-writer (replaces single DuckDB)
# ══════════════════════════════════════════════════════════════════════════════


class PgCheckpointManager:
    """
    PostgreSQL-backed checkpoint manager.

    Replaces the DuckDB single-writer bottleneck with PG MVCC:
    100+ concurrent workers can write checkpoints simultaneously.
    """

    def __init__(self, pool, config: PgConfig):
        self._pool = pool
        self._table = f"{config.shared_schema}.{config.checkpoint_table}"

    def _key(self, phase: str, account_id: str, unit_id: str) -> str:
        return f"{phase}:{account_id}:{unit_id}"

    async def is_completed(self, phase: str, account_id: str, unit_id: str) -> bool:
        key = self._key(phase, account_id, unit_id)
        async with self._pool.acquire() as conn:
            row = await conn.fetchval(
                f"SELECT status FROM {self._table} WHERE checkpoint_key = $1", key,
            )
        return row == "completed"

    async def mark_started(self, phase: str, account_id: str, run_id: str, unit_id: str):
        await self._write(phase, account_id, run_id, unit_id, "in_progress")

    async def mark_completed(
        self, phase: str, account_id: str, run_id: str, unit_id: str,
        artifact_path: str | None = None, metadata: dict | None = None,
    ):
        await self._write(
            phase, account_id, run_id, unit_id, "completed",
            artifact_path=artifact_path, metadata=metadata,
        )

    async def mark_failed(
        self, phase: str, account_id: str, run_id: str, unit_id: str, error: str,
    ):
        await self._write(phase, account_id, run_id, unit_id, "failed", error=error)

    async def phase_completed(self, phase: str, account_id: str) -> bool:
        return await self.is_completed(phase, account_id, "__phase__")

    async def mark_phase_completed(
        self, phase: str, account_id: str, run_id: str, metadata: dict | None = None,
    ):
        await self.mark_completed(phase, account_id, run_id, "__phase__", metadata=metadata)

    async def get_incomplete(self, phase: str, account_id: str) -> list[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT unit_id FROM {self._table} WHERE phase = $1 AND account_id = $2 AND status != 'completed'",
                phase, account_id,
            )
        return [r["unit_id"] for r in rows]

    async def _write(
        self, phase: str, account_id: str, run_id: str, unit_id: str,
        status: str, artifact_path: str | None = None,
        metadata: dict | None = None, error: str | None = None,
    ):
        key = self._key(phase, account_id, unit_id)
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table}
                (checkpoint_key, phase, account_id, run_id, unit_id, status,
                 artifact_path, metadata, error)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
                ON CONFLICT (checkpoint_key) DO UPDATE SET
                    status = EXCLUDED.status,
                    artifact_path = EXCLUDED.artifact_path,
                    metadata = EXCLUDED.metadata,
                    error = EXCLUDED.error,
                    created_at = NOW()
            """, key, phase, account_id, run_id, unit_id, status,
                artifact_path, json.dumps(metadata or {}), error)

    def close(self):
        """No-op — pool is managed by PgStoreManager."""
        pass


# ══════════════════════════════════════════════════════════════════════════════
# §5  PG FEEDBACK STORE — Replaces per-tenant DuckDB sidecar
# ══════════════════════════════════════════════════════════════════════════════


class PgFeedbackStore:
    """
    Account/tenant-scoped feedback store backed by PostgreSQL.

    Replaces the per-tenant DuckDB sidecar with PG tables in the
    account schema. Supports async writes (no contention).
    """

    def __init__(self, pool, schema: str, tenant_id: str):
        self._pool = pool
        self._schema = schema
        self.tenant_id = tenant_id

    def _table(self, name: str) -> str:
        return f"{self._schema}.{name}"

    async def record_feedback(
        self, user_id: str, entity_type: str, entity_id: str,
        action: str, old_value: dict | None = None,
        new_value: dict | None = None, context: dict | None = None,
        source_doc_id: str | None = None, session_id: str | None = None,
    ) -> str:
        event_id = f"fb_{uuid.uuid4().hex[:12]}"
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self._table('feedback_events')}
                (event_id, tenant_id, user_id, entity_type, entity_id, action,
                 context, old_value, new_value, source_doc_id, session_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9::jsonb, $10, $11)
            """, event_id, self.tenant_id, user_id, entity_type, entity_id,
                action, json.dumps(context or {}),
                json.dumps(old_value or {}), json.dumps(new_value or {}),
                source_doc_id, session_id)

            # Update weight
            await self._update_weight(conn, entity_type, entity_id, action)

        return event_id

    async def _update_weight(self, conn, entity_type: str, entity_id: str, action: str):
        adjustment = {"correct": 0.3, "reject": -0.2, "approve": 0.1,
                       "refine": 0.15, "merge": 0.2, "split": 0.2}.get(action, 0.0)
        weight_id = f"pw_{self.tenant_id}_{entity_type}_{entity_id}"
        await conn.execute(f"""
            INSERT INTO {self._table('personalization_weights')}
            (weight_id, tenant_id, entity_type, entity_id, weight, adjustment, feedback_count, last_updated)
            VALUES ($1, $2, $3, $4, $5, $6, 1, NOW())
            ON CONFLICT (weight_id) DO UPDATE SET
                weight = {self._table('personalization_weights')}.weight + EXCLUDED.adjustment,
                adjustment = EXCLUDED.adjustment,
                feedback_count = {self._table('personalization_weights')}.feedback_count + 1,
                last_updated = NOW()
        """, weight_id, self.tenant_id, entity_type, entity_id,
            1.0 + adjustment, adjustment)

    async def get_feedback_events(
        self, entity_type: str | None = None, entity_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        conditions = [f"tenant_id = $1"]
        params: list[Any] = [self.tenant_id]
        idx = 2
        if entity_type:
            conditions.append(f"entity_type = ${idx}")
            params.append(entity_type)
            idx += 1
        if entity_id:
            conditions.append(f"entity_id = ${idx}")
            params.append(entity_id)
            idx += 1
        where = " AND ".join(conditions)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self._table('feedback_events')} WHERE {where} ORDER BY created_at DESC LIMIT {limit}",
                *params,
            )
        return [dict(r) for r in rows]

    async def get_weight(self, entity_type: str, entity_id: str) -> float:
        async with self._pool.acquire() as conn:
            row = await conn.fetchval(f"""
                SELECT weight FROM {self._table('personalization_weights')}
                WHERE tenant_id = $1 AND entity_type = $2 AND entity_id = $3
            """, self.tenant_id, entity_type, entity_id)
        return float(row) if row else 1.0

    def close(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# §6  SETUP UTILITIES — Extension installation + health checks
# ══════════════════════════════════════════════════════════════════════════════


async def check_pg_extensions(dsn: str) -> dict[str, bool]:
    """Check which required extensions are installed."""
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            "SELECT extname FROM pg_extension"
        )
        installed = {r["extname"] for r in rows}
    finally:
        await conn.close()

    required = {"vector": False, "pg_trgm": False, "btree_gin": False}
    optional = {"pg_search": False}

    for ext in required:
        required[ext] = ext in installed
    for ext in optional:
        optional[ext] = ext in installed

    return {**required, **optional}


async def setup_pg_database(dsn: str, config: PgConfig | None = None) -> PgStoreManager:
    """
    One-shot setup: create pool, install extensions, return manager.

    Usage:
        mgr = await setup_pg_database("postgresql://localhost:5432/clustering_v2")
        store = await mgr.get_store("acct_001")
    """
    cfg = config or PgConfig(dsn=dsn)
    mgr = PgStoreManager(cfg)
    await mgr.initialize()
    return mgr


async def health_check(pool) -> dict:
    """Quick health check: connection, extensions, table counts."""
    async with pool.acquire() as conn:
        # PG version
        version = await conn.fetchval("SELECT version()")
        # Extensions
        exts = await conn.fetch("SELECT extname, extversion FROM pg_extension")
        # Schema count
        n_schemas = await conn.fetchval(
            "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name LIKE 'acct_%'"
        )
        # Checkpoint count
        n_checkpoints = await conn.fetchval(
            "SELECT COUNT(*) FROM checkpoints"
        )
    return {
        "pg_version": version,
        "extensions": {r["extname"]: r["extversion"] for r in exts},
        "n_account_schemas": n_schemas,
        "n_checkpoints": n_checkpoints,
        "status": "healthy",
    }
