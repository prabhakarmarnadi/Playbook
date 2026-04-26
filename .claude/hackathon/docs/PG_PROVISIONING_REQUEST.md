# PostgreSQL Provisioning Request — Clustering V2 Pipeline

**Team:** Agreement Intelligence  
**Date:** April 20, 2026  
**Priority:** High — blocking production scale-out (1M docs / 10K accounts)  
**Contact:** Aaditya Srivathsan

---

## 1. What We Need

A single PostgreSQL cluster to replace our current per-account DuckDB + LanceDB file storage. This is the backend for our agreement clustering, field discovery, and extraction pipeline.

## 2. PostgreSQL Server Requirements

| Requirement | Spec |
|-------------|------|
| **PostgreSQL version** | **16+** (must support GENERATED ALWAYS AS ... STORED columns) |
| **Compute** | 8+ vCPUs, 64 GB RAM minimum (embeddings are 768-dim float32 vectors) |
| **Storage** | 2 TB SSD to start, expandable to 10+ TB (grows ~200 GB per 1K accounts) |
| **OS** | Ubuntu 22.04+ or RHEL 9+ |
| **High Availability** | Streaming replication with at least 1 read replica (analytical queries go here) |
| **Connection limit** | `max_connections = 200` minimum (we pool via pgbouncer/asyncpg) |
| **Shared memory** | `shared_buffers = 16GB`, `work_mem = 256MB`, `maintenance_work_mem = 2GB` |
| **Max WAL size** | `max_wal_size = 4GB` (heavy batch inserts during Phase 0 ingest) |

## 3. Required Extensions

### 3a. pgvector (REQUIRED — blocks all vector search)

We store 768-dimensional float32 embeddings and run ANN (approximate nearest neighbor) search on them. pgvector provides the `vector` data type and HNSW indexes.

**Version:** 0.7.0 or later  
**Install:**
```bash
# From source (if not available via package manager):
cd /tmp
git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Then in psql as superuser:
CREATE EXTENSION IF NOT EXISTS vector;
```

**Or via apt (Ubuntu):**
```bash
sudo apt install postgresql-16-pgvector
# then:
CREATE EXTENSION IF NOT EXISTS vector;
```

**Verification:**
```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
-- Should return one row with extversion >= 0.7.0

-- Quick smoke test:
CREATE TABLE test_vec (id serial, embedding vector(768));
INSERT INTO test_vec (embedding) VALUES ('[' || array_to_string(array_agg(random()::float), ',') || ']' FROM generate_series(1, 768));
CREATE INDEX ON test_vec USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
SELECT * FROM test_vec ORDER BY embedding <=> (SELECT embedding FROM test_vec LIMIT 1) LIMIT 5;
DROP TABLE test_vec;
```

### 3b. pg_trgm (REQUIRED — built-in, just needs enabling)

Used for trigram similarity on text fields (fuzzy matching of clause titles, field names).

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

This ships with PostgreSQL — no additional installation needed.

### 3c. btree_gin (REQUIRED — built-in, just needs enabling)

Enables GIN index support for standard data types alongside our tsvector full-text indexes.

```sql
CREATE EXTENSION IF NOT EXISTS btree_gin;
```

Also ships with PostgreSQL core — no additional installation.

### 3d. pg_search / ParadeDB (OPTIONAL — nice to have)

Native BM25 scoring for full-text search. If not available, we fall back to PostgreSQL's built-in `tsvector` + `ts_rank_cd()` which works fine.

**Only install if ParadeDB packages are available in your environment:**
```bash
# ParadeDB provides a custom Postgres build or extension packages
# See: https://docs.paradedb.com/deploy/self-hosted
```

```sql
CREATE EXTENSION IF NOT EXISTS pg_search;
```

**We gracefully fall back if this is not present. Do not block on this.**

## 4. Database & Role Setup

```sql
-- Create the database
CREATE DATABASE clustering_v2
    ENCODING 'UTF8'
    LC_COLLATE 'en_US.UTF-8'
    LC_CTYPE 'en_US.UTF-8';

-- Create application role (least privilege)
CREATE ROLE clustering_app LOGIN PASSWORD '<generated>';

-- Grant schema creation (we auto-provision per-account schemas at runtime)
GRANT CREATE ON DATABASE clustering_v2 TO clustering_app;

-- Connect to the DB and install extensions (requires superuser)
\c clustering_v2

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Grant extension usage to app role
GRANT USAGE ON SCHEMA public TO clustering_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO clustering_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL PRIVILEGES ON TABLES TO clustering_app;
```

## 5. What the App Creates at Runtime

The application handles all schema/table/index creation automatically via `PgStoreManager.initialize()`. Specifically:

**Shared tables (public schema):**
- `public.checkpoints` — pipeline checkpoint tracking (multi-writer via MVCC)
- `public.pipeline_runs` — run status log

**Per-account schemas (created on first access):**
```
acct_{account_id}/
├── agreements         (relational, B-tree on domain_id)
├── chunks             (relational + vector(768) + tsvector GENERATED column)
│   ├── HNSW index     on embedding (vector_cosine_ops, m=16, ef=128)
│   ├── GIN index      on tsv (tsvector) for BM25 full-text search
│   └── B-tree indexes on agreement_id, clause_id
├── clauses            (relational, B-tree on agreement_id)
├── clusters           (relational + vector(768) centroid, JSONB keywords)
├── cluster_assignments (composite PK, B-tree on cluster_id)
├── field_definitions  (relational, B-tree on cluster_id)
├── extractions        (relational, compound B-tree on agreement_id + field_id)
├── documents          (vector(768) + HNSW index for doc-level search)
├── cluster_centroids  (vector(768))
├── intent_types       (relational)
├── clause_intents     (relational, B-tree on agreement_id, clause_id)
├── cluster_layers     (composite PK)
├── cluster_layer_meta (composite PK)
├── feedback_events    (GIN on entity_type + entity_id)
├── personalization_weights
├── virtual_clusters   (vector(768) centroid)
├── virtual_assignments
├── intent_overrides
└── tenant_config
```

**The app role needs:** `CREATE` on database (for schemas), `ALL` on tables within schemas it creates.

## 6. Connection Details We Need Back

Please provide:
1. **DSN:** `postgresql://<user>:<password>@<host>:<port>/clustering_v2`
2. **Read replica DSN** (if separate): for analytical/dashboard queries
3. **Whether pgbouncer is in front** (we use asyncpg connection pooling; if pgbouncer is present, we'll set `statement_cache_size=0`)
4. **Installed extension versions** — output of:
   ```sql
   SELECT extname, extversion FROM pg_extension ORDER BY extname;
   ```

## 7. Sizing Estimates

| Scale | Accounts | Schemas | Rows (chunks) | Vector index size | Total disk |
|-------|----------|---------|---------------|-------------------|-----------|
| Dev/test | 10 | 10 | 100K | ~500 MB | 5 GB |
| Pilot | 100 | 100 | 1M | ~5 GB | 50 GB |
| Production | 10K | 10K | 100M | ~500 GB | 2-5 TB |

Each vector index entry = 768 dims × 4 bytes + HNSW graph overhead ≈ 4 KB/vector.

## 8. Key PostgreSQL Config Tweaks

These help with our workload (large batch inserts + high-concurrency ANN search):

```ini
# postgresql.conf additions/overrides:
shared_buffers = 16GB
work_mem = 256MB
maintenance_work_mem = 2GB
max_wal_size = 4GB
effective_cache_size = 48GB
random_page_cost = 1.1          # SSD storage
max_parallel_workers_per_gather = 4
max_parallel_maintenance_workers = 4

# pgvector-specific (added to postgresql.conf or SET per-session):
# hnsw.ef_search = 100          # we SET this per-query via SET LOCAL
```

---

**Summary: We need PostgreSQL 16+ with pgvector 0.7+, pg_trgm, and btree_gin enabled. The app creates and manages all schemas and tables automatically. Please provide the DSN and extension confirmation once provisioned.**
