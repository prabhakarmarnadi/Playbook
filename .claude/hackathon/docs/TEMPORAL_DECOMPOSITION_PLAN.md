# Temporal Decomposition Plan — Minimal Pipeline

> **Audience**: An implementation agent (human or AI) tasked with converting the monolithic
> `core/minimal_pipeline.py` into a Temporal-orchestrated microservice architecture.
>
> **Prerequisites**: Read the architectural breakdown comment at the top of
> `core/minimal_pipeline.py` (sections 1–12) before starting. That comment contains the
> complete function index, config inventory, data-store map, and module groupings you'll
> reference throughout this plan.

---

## Table of Contents

1. [Context & Goals](#1-context--goals)
2. [Source File Map](#2-source-file-map)
3. [Monolith → Temporal Concept Mapping](#3-monolith--temporal-concept-mapping)
4. [Workflow Topology](#4-workflow-topology)
5. [Task Queues & Worker Configuration](#5-task-queues--worker-configuration)
6. [Activity Definitions](#6-activity-definitions)
7. [Data Passing Strategy](#7-data-passing-strategy)
8. [Temporal Cluster Topology](#8-temporal-cluster-topology)
9. [Rate Limiting & Concurrency](#9-rate-limiting--concurrency)
10. [Inter-Service Data Contracts](#10-inter-service-data-contracts)
11. [The Real-Time Assignment Path](#11-the-real-time-assignment-path)
12. [Implementation Phases](#12-implementation-phases)
13. [File Structure (Target)](#13-file-structure-target)
14. [Testing Strategy](#14-testing-strategy)
15. [Failure Modes & Mitigations](#15-failure-modes--mitigations)

---

## 1. Context & Goals

### What the monolith does

`core/minimal_pipeline.py` (~3100 lines) is a single-file async Python pipeline that:

1. Parses a directory of PDF legal agreements
2. Chunks them into 3 types (embedding, extraction, clause) and embeds with Nomic (768-d CUDA)
3. Runs parallel EVoC clustering at two levels (macro documents, micro chunks) with optional guided clustering, multi-dimensional signals, and soft probabilistic memberships
4. Discovers field schemas per cluster via RLM V3 (agentic REPL with 12 tools) and extracts structured values via agentic RAG — gated by a confidence-sampling mechanism
5. Optionally discovers legal intents per clause
6. Initialises an incremental assigner for real-time single-document intake
7. Persists everything to DuckDB (main + feedback sidecar) and LanceDB (vectors)

### Why Temporal

| Current limitation | Temporal solution |
|---|---|
| No crash recovery — full restart on failure | Durable execution: replays from last completed activity |
| `asyncio.Semaphore` for all concurrency | Per-queue worker limits + distributed rate limiting |
| No per-cluster retry isolation | Each activity retries independently |
| No stage-level observability | Temporal UI: workflow state, activity history, timings |
| Single DuckDB connection shared across stages | Each activity opens its own; workflows pass only references |
| Can't independently scale GPU vs LLM workloads | Separate task queues with different worker pools |
| `uuid4` IDs — no idempotency | Workflow ID = `{account_id}/{run_id}` — deterministic dedup |

### Non-goals

- Rewriting the core algorithms (EVoC, RLM V3, RAG extraction). These stay as-is inside activity implementations.
- Changing the data stores (DuckDB, LanceDB). Activities read/write them directly.
- Building a REST API layer. That's a separate concern above the Temporal client.

---

## 2. Source File Map

All paths relative to `hackathon/`.

| File | Role | Temporal Relevance |
|---|---|---|
| `core/minimal_pipeline.py` | The monolith — contains everything | Source of truth; to be decomposed |
| `core/store.py` | `ClusteringStore` — DuckDB wrapper | Shared state; each activity instantiates its own connection |
| `core/lancedb_store.py` | `LanceVectorStore` — vector persistence | Written by embedding activity, read by RAG extraction |
| `core/nomic_embedder.py` | `NomicEmbedder` — GPU embedding (768-d Nomic) | GPU-affinity: runs only on `embedding` + `clustering` workers |
| `core/embedder.py` | `Embedder` singleton (used by RAG + assigner) | Used by extraction + assignment activities |
| `core/pdf_parser.py` | PDF parsing + caching | Used by ingestion activity |
| `core/chunker.py` | 3-type chunking + clause header extraction | Used by ingestion activity |
| `core/field_features.py` | `FieldFeatureExtractor` (pattern/keyword matching) | Used by ingestion activity (optional) |
| `core/macro_clusterer.py` | `get_field_summaries` — document summaries | Used by embedding activity |
| `core/evoc_pipeline.py` | EVoC clustering (HDBSCAN + Optuna) | Clustering activity core logic |
| `core/topic_merger.py` | `HierarchicalTopicMerger` | Used by clustering activity |
| `core/rlm_v3.py` | RLM V3 agentic REPL (12 tools) | Discovery activity core logic |
| `core/rag_extractor.py` | `RAGExtractor` (agentic RAG + retries + reviewer) | Extraction activity core logic |
| `core/intent_extractor.py` | `IntentRegistry` | Intent activity core logic |
| `core/feedback_store.py` | `FeedbackStore` — sidecar DuckDB (8 action types) | Feedback service data layer |
| `core/personalization.py` | `PersonalizationEngine` — context from feedback | Feedback service logic layer |
| `core/cluster_assigner.py` | `IncrementalClusterAssigner` (3-tier routing) | Assignment service core logic |

### Key data structures defined in `minimal_pipeline.py`

These must be extracted into a shared `models/` package:

| Structure | Location (approx line) | Description |
|---|---|---|
| `ClusterProfile` | ~45–125 | The "agreement genome" — 14-field dataclass with `to_dict`/`from_dict` |
| `MinimalPipelineConfig` | ~665–730 | 35-field config dataclass — all tunables |
| `AsyncLLMClient` | ~735–800 | Rate-limited Azure OpenAI wrapper |

---

## 3. Monolith → Temporal Concept Mapping

| Monolith Concept | Temporal Equivalent |
|---|---|
| `run_minimal_pipeline()` | `PipelineWorkflow` (top-level workflow) |
| Stage 1–3, 5–6 blocks | `executeActivity()` calls within the workflow |
| Stage 4 `asyncio.gather()` over clusters | `ExtractionFanOutWorkflow` (child workflow) |
| `_process_cluster()` | `ProcessCluster` activity |
| `asyncio.Semaphore(CLUSTER_CONCURRENCY)` | Worker-level `max_concurrent_activity_execution_size` |
| `AsyncLLMClient` semaphore | Distributed rate limiter (Redis token bucket or Temporal activity-level) |
| `assign_new_documents()` | `AssignDocumentWorkflow` (separate namespace) |
| `MinimalPipelineConfig` | Workflow input payload (serialized dataclass) |
| `_StageTimer` | Temporal activity timing (built into history) |
| `progress()` callback | Temporal workflow `upsert_search_attributes()` or custom query handler |
| `pipeline_stats` dict | Workflow result payload |

---

## 4. Workflow Topology

### A. Batch Pipeline Workflow

```
PipelineWorkflow(pdf_dir, account_id, config)
│
├─ 1. executeActivity("IngestPDFs")                    queue: ingestion
│     input:  {pdf_dir, account_id, max_docs, chunker_config, field_features_config}
│     output: {run_id, n_docs, n_chunks, chunk_manifest_key}
│
├─ 2. executeActivity("EmbedChunks")                   queue: embedding
│     input:  {run_id, account_id, device, chunk_manifest_key}
│     output: {n_chunk_embeddings, n_summary_embeddings, embedding_ref}
│
├─ 3. executeActivity("RunClustering")                 queue: clustering
│     input:  {run_id, account_id, embedding_ref, clustering_config}
│     output: {n_domains, n_clusters, cluster_ids[], centroids_ref, soft_memberships_ref}
│
├─ 4. executeChildWorkflow("ExtractionFanOutWorkflow")  queue: discovery-extraction
│     input:  {run_id, account_id, cluster_ids[], config, personalization_ctx}
│     │
│     ├─ for each cluster_id (parallel, bounded):
│     │   executeActivity("ProcessCluster")
│     │     input:  {cluster_id, run_id, account_id, config, personalization_ctx}
│     │     output: {fields, extractions, profile, confidence_gate}
│     │
│     output: {total_fields, total_extractions, profiles[], stats}
│
├─ 5. [conditional: config.run_intents]
│     executeActivity("DiscoverIntents")               queue: intent
│     input:  {run_id, account_id}
│     output: {total_clauses, total_intents}
│
├─ 6. executeActivity("InitAssigner")                  queue: assignment
│     input:  {account_id, personalization_ctx}
│     output: {n_centroids, thresholds}
│
└─ 7. executeActivity("PersistManifest")               queue: ingestion
      input:  {run_id, account_id, all_stage_stats}
      output: {manifest_path}
```

### B. Real-Time Assignment Workflow

```
AssignDocumentWorkflow(pdf_path, account_id, config)
│
├─ 1. executeActivity("ParseSinglePDF")               queue: ingestion
│     input:  {pdf_path}
│     output: {text, agreement_id, filename}
│
├─ 2. executeActivity("EmbedText")                     queue: embedding
│     input:  {text, device}
│     output: {embedding}  # (768,) float32
│
├─ 3. executeActivity("AssignToCluster")               queue: assignment
│     input:  {embedding, account_id, personalization_ctx}
│     output: {cluster_id, label, confidence, assignment_type, top_matches}
│
└─ 4. [conditional: assignment_type == "novel"]
      signal PipelineWorkflow or NovelBufferWorkflow: "novel_document_added"
```

---

## 5. Task Queues & Worker Configuration

### Queue Definitions

| Task Queue | Module(s) | Worker Type | Scaling Axis | `max_concurrent_activity_execution_size` |
|---|---|---|---|---|
| `ingestion` | A: Ingestion | CPU, high-memory | Document throughput | 4 |
| `embedding` | B: Embedding | **GPU** (CUDA) | GPU count, batch size | 2 |
| `clustering` | C: Clustering | CPU-heavy | Core count | 1 (global sync stage) |
| `discovery-extraction` | D: Discovery + Extraction | CPU, LLM network-bound | LLM rate limit | 3 (matches current `CLUSTER_CONCURRENCY`) |
| `intent` | E: Intent | CPU, LLM-bound | LLM rate limit | 4 |
| `assignment` | F: Assignment | CPU, low-latency | Request rate | 8 |
| `feedback` | G: Feedback | CPU, low-latency | Tenant count | 2 |

### Worker Process Configuration

```python
# GPU worker (polls: embedding, clustering)
gpu_worker = Worker(
    client=temporal_client,
    task_queue="embedding",
    workflows=[],  # GPU workers only run activities
    activities=[embed_chunks, run_clustering],
    max_concurrent_activities=2,
    max_task_queue_activities_per_second=10,
)

# LLM worker (polls: discovery-extraction, intent)
llm_worker = Worker(
    client=temporal_client,
    task_queue="discovery-extraction",
    activities=[process_cluster, discover_intents],
    max_concurrent_activities=3,  # replaces asyncio.Semaphore(CLUSTER_CONCURRENCY)
    max_task_queue_activities_per_second=5,  # global LLM rate limit
)

# Assignment worker (polls: assignment) — pre-warmed embedder + centroid cache
assignment_worker = Worker(
    client=temporal_client,
    task_queue="assignment",
    activities=[assign_to_cluster, parse_single_pdf, embed_text],
    max_concurrent_activities=8,
)
```

### Worker → Queue Polling Matrix

A single worker process can poll **multiple queues** if it has the right hardware:

```
GPU Instance:
  └─ Worker 1: polls [embedding, clustering]

CPU Instance (LLM-bound):
  └─ Worker 2: polls [discovery-extraction, intent]

CPU Instance (general):
  └─ Worker 3: polls [ingestion, feedback]

CPU Instance (low-latency, pre-warmed):
  └─ Worker 4: polls [assignment]
```

---

## 6. Activity Definitions

### 6.1 IngestPDFs

```python
@activity.defn
async def ingest_pdfs(input: IngestInput) -> IngestOutput:
    """Stage 1+2a: Parse PDFs + generate 3 chunk types + field features.

    Maps to: run_minimal_pipeline lines ~1535–1720 (Stage 1 + Stage 2 chunking portion)

    Monolith functions used:
      - core.pdf_parser.parse_pdf_directory / load_parsed_agreements
      - core.chunker.generate_typed_chunks, extract_clause_header
      - core.field_features.FieldFeatureExtractor.batch_extract_features

    Input:
      pdf_dir:              str — filesystem path to PDF directory
      account_id:           str
      max_docs:             int | None
      chunker_config:       dict — {min_chunk_chars, max_chunk_chars, extraction_chunk_chars, max_chunks_per_doc}
      field_features_config: dict — {enable, legacy_fields_dir}

    Output:
      run_id:            str — generated UUID-based run identifier
      n_docs:            int — number of valid documents
      n_embedding_chunks: int
      n_extraction_chunks: int
      n_clauses:         int
      chunk_manifest_key: str — reference key for retrieving chunks from DuckDB
        (run_id is sufficient since chunks are stored with agreement_id linkage)

    Heartbeat: reports parsed doc count every 10 documents.
    Timeout: start_to_close = 10 minutes.
    Retry: max_attempts=2, backoff=10s (PDF parsing is mostly deterministic).
    """
```

### 6.2 EmbedChunks

```python
@activity.defn
async def embed_chunks(input: EmbedInput) -> EmbedOutput:
    """Stage 2b+2c: Nomic embedding + summary generation + persistence.

    Maps to: run_minimal_pipeline lines ~1700–1770 (embedding + persist portion)

    Monolith functions used:
      - core.nomic_embedder.NomicEmbedder.embed_for_clustering
      - core.macro_clusterer.get_field_summaries
      - core.store.ClusteringStore (insert_chunk, insert_clause, insert_agreement)
      - core.lancedb_store.LanceVectorStore (upsert_chunks, upsert_documents)

    Input:
      run_id:               str
      account_id:           str
      device:               str — "cuda" or "cpu"
      chunk_manifest_key:   str — used to query chunks from DuckDB

    Output:
      n_chunk_embeddings:   int
      n_summary_embeddings: int
      embedding_ref:        str — reference for downstream (just run_id + account_id)

    MUST run on GPU worker (task queue: embedding).
    Heartbeat: reports embedding batch progress.
    Timeout: start_to_close = 5 minutes.
    Retry: max_attempts=2 (GPU OOM is possible; retry with smaller batch on second attempt).
    """
```

### 6.3 RunClustering

```python
@activity.defn
async def run_clustering(input: ClusterInput) -> ClusterOutput:
    """Stage 3: Macro + micro EVoC clustering with all transforms.

    Maps to: run_minimal_pipeline lines ~1770–1960 (full Stage 3 block)

    Monolith functions used:
      - core.evoc_pipeline.evoc_cluster, evoc_optimize, _select_best_layer, _llm_label_clusters
      - compute_guided_embeddings()          [from minimal_pipeline.py]
      - compute_multidimensional_embeddings() [from minimal_pipeline.py]
      - compute_soft_memberships()           [from minimal_pipeline.py]
      - identify_low_confidence_chunks()     [from minimal_pipeline.py]
      - extract_cluster_keywords()           [from minimal_pipeline.py]

    Input:
      run_id:            str
      account_id:        str
      embedding_ref:     str
      clustering_config: dict — {use_optuna, optuna_trials, evoc_noise_level,
                                  guided_keywords, guided_alpha,
                                  enable_multidim_signals, field_overlap_weight,
                                  obligation_weight, enable_soft_membership,
                                  soft_membership_k, low_confidence_threshold}

    Output:
      n_domains:          int
      n_clusters:         int
      cluster_ids:        list[str]
      centroids_ref:      str — reference for downstream centroid retrieval
      soft_memberships_ref: str — reference (or inline if small enough)

    NOTE: This is globally synchronous — needs all embeddings in memory at once.
    Cannot be parallelised at the document level.
    Timeout: start_to_close = 15 minutes (Optuna can be slow with many trials).
    Retry: max_attempts=2.
    """
```

### 6.4 ExtractionFanOutWorkflow + ProcessCluster

```python
@workflow.defn
class ExtractionFanOutWorkflow:
    """Stage 4: Fan-out discovery + extraction across all clusters.

    Maps to: run_minimal_pipeline lines ~1960–2110 (Stage 4 block)

    This is a CHILD WORKFLOW, not an activity, because it orchestrates
    parallel activities with bounded concurrency.

    The monolith uses asyncio.gather() with a Semaphore(CLUSTER_CONCURRENCY).
    In Temporal, each cluster becomes an independent activity invocation.
    Concurrency is controlled by the worker's max_concurrent_activities setting
    on the discovery-extraction task queue.

    Input:
      run_id:              str
      account_id:          str
      cluster_ids:         list[str]
      config:              dict — discovery + extraction settings
      personalization_ctx: dict | None — from FeedbackStore

    Output:
      total_fields:      int
      total_extractions: int
      n_profiles:        int
      per_cluster_stats: list[dict]
    """

    @workflow.run
    async def run(self, input):
        # Fan out: one activity per cluster
        # Temporal bounds concurrency via worker config, not in-workflow semaphores
        futures = []
        for cluster_id in input.cluster_ids:
            futures.append(
                workflow.execute_activity(
                    process_cluster,
                    ProcessClusterInput(
                        cluster_id=cluster_id,
                        run_id=input.run_id,
                        account_id=input.account_id,
                        config=input.config,
                        personalization_ctx=input.personalization_ctx,
                    ),
                    task_queue="discovery-extraction",
                    start_to_close_timeout=timedelta(minutes=10),
                    retry_policy=RetryPolicy(
                        maximum_attempts=3,
                        initial_interval=timedelta(seconds=5),
                        backoff_coefficient=2.0,
                        non_retryable_error_types=["ClusterTooSmallError"],
                    ),
                )
            )
        results = await asyncio.gather(*futures, return_exceptions=True)
        # Aggregate — same logic as current monolith's result collection loop
        ...


@activity.defn
async def process_cluster(input: ProcessClusterInput) -> ProcessClusterOutput:
    """Process one cluster: discovery → confidence gate → RAG extraction.

    Maps to: _process_cluster() in minimal_pipeline.py (lines ~978–1195)

    Monolith functions used:
      - _diversity_sample_chunks()
      - _discover_and_extract_rlm_v3() or _discover_fields_rlm_v3() + fallback
      - _cochran_sample_size(), _compute_extraction_confidence()
      - _run_rag_extraction()
      - _build_cluster_profile(), _build_rlm_context_section()
      - _normalize_field()

    The 3-phase logic stays INSIDE this single activity:
      Phase 1: RLM V3 discovery (or standard fallback)
      Phase 2: Confidence-gated example-set extraction
      Phase 3: Full RAG extraction (if gate passes)

    Reason for keeping phases bundled: they share in-memory state (chunk texts,
    field definitions, rlm_meta) and splitting them into 3 activities would require
    serializing intermediate state with no real benefit — each phase is fast
    relative to the LLM calls within it.

    Input:
      cluster_id:          str
      run_id:              str
      account_id:          str
      config:              dict
      personalization_ctx: dict | None

    Output:
      cluster_id:      str
      fields:          int — number of fields discovered
      extractions:     int — number of values extracted
      confidence_gate: dict — gate metrics and pass/fail
      profile:         dict — serialized ClusterProfile

    Heartbeat: reports phase transitions (discovery → gate → extraction).
    Timeout: start_to_close = 10 minutes per cluster.
    Retry: max_attempts=3 (LLM calls can timeout; RLM V3 can hit iteration limits).
    Non-retryable: ClusterTooSmallError (< min_cluster_chunks).
    """
```

### 6.5 DiscoverIntents

```python
@activity.defn
async def discover_intents(input: IntentInput) -> IntentOutput:
    """Stage 5: Intent discovery across all clauses.

    Maps to: _process_intents() in minimal_pipeline.py (lines ~1358–1430)

    Input:  {run_id, account_id, min_clause_chars, match_threshold}
    Output: {total_clauses, total_intents}

    Timeout: start_to_close = 15 minutes (many LLM calls).
    Retry: max_attempts=2.
    """
```

### 6.6 InitAssigner

```python
@activity.defn
async def init_assigner(input: AssignerInput) -> AssignerOutput:
    """Stage 6: Load centroids into IncrementalClusterAssigner.

    Maps to: Stage 6 block in run_minimal_pipeline (lines ~2120–2165)

    Input:  {account_id, personalization_ctx}
    Output: {n_centroids, high_threshold, tentative_threshold}

    Lightweight activity — just loads centroids from DuckDB.
    Timeout: start_to_close = 30 seconds.
    Retry: max_attempts=2.
    """
```

---

## 7. Data Passing Strategy

### The 2MB Rule

Temporal payloads must be < 2MB (configurable, but default). This constrains what travels through workflow history vs. external storage.

| Data | Size | Passes through Temporal? | Storage |
|---|---|---|---|
| Config (`MinimalPipelineConfig`) | ~2KB | **Yes** — workflow input | Serialized dataclass |
| `run_id`, `account_id`, `cluster_ids` | bytes | **Yes** — activity inputs/outputs | Plain strings |
| `PersonalizationContext` | ~5KB | **Yes** — activity input | Serialized dict |
| Per-cluster stats / profile summary | ~10KB | **Yes** — activity output | Dict |
| `ClusterProfile` (full, per cluster) | ~50KB | **Yes** — in activity output, aggregated in child workflow | Serialized dict |
| Chunk texts (all) | 10–500MB | **No** | DuckDB (queried by run_id) |
| Embeddings (`N×768 float32`) | 10–200MB | **No** | LanceDB + DuckDB (queried by run_id) |
| Raw PDF content | arbitrary | **No** | Filesystem / blob storage |
| Pipeline manifest | ~20KB | **Yes** — final workflow output | Also persisted as JSON artifact |

### Reference Pattern

Activities receive **references** (run_id, account_id, cluster_id) and query the data stores directly. They never receive large arrays or text blobs in their input payloads.

```python
# GOOD: pass a reference
embed_input = EmbedInput(run_id="run_abc", account_id="acct_001", device="cuda")

# BAD: pass the actual data
embed_input = EmbedInput(chunks=[...10000 chunks...], embeddings=np.array(...))
```

### Shared Data Store Access

Each activity instantiates its own `ClusteringStore(db_path)` connection. The `db_path` is deterministic from `account_id`:

```python
db_path = DATA_DIR / f"{account_id}.duckdb"
store = ClusteringStore(db_path)  # each activity opens its own connection
```

Since DuckDB supports concurrent readers and serialized writers, this works for moderate concurrency. At scale, consider:
- Read replicas for extraction activities (read-heavy)
- Write batching for ingestion activities (write-heavy)
- Or migrating to PostgreSQL with connection pooling

---

## 8. Temporal Cluster Topology

### Two Namespaces, One Cluster

```
┌─────────────────────────────────────────────────────────────────┐
│ Temporal Server (self-hosted or Temporal Cloud)                  │
│                                                                 │
│  ┌─ Namespace: "extraction-batch" ───────────────────────────┐  │
│  │                                                           │  │
│  │  Workflows:                                               │  │
│  │    PipelineWorkflow                                       │  │
│  │    ExtractionFanOutWorkflow                               │  │
│  │                                                           │  │
│  │  Task Queues:                                             │  │
│  │    ingestion, embedding, clustering,                      │  │
│  │    discovery-extraction, intent                           │  │
│  │                                                           │  │
│  │  Workers:                                                 │  │
│  │    2× GPU (embedding + clustering)                        │  │
│  │    4× CPU (ingestion + discovery-extraction + intent)     │  │
│  │                                                           │  │
│  │  Retention: 30 days                                       │  │
│  │  Search attributes:                                       │  │
│  │    account_id (Keyword), run_id (Keyword),                │  │
│  │    stage (Keyword), n_clusters (Int), n_extractions (Int) │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Namespace: "extraction-realtime" ────────────────────────┐  │
│  │                                                           │  │
│  │  Workflows:                                               │  │
│  │    AssignDocumentWorkflow                                 │  │
│  │    NovelBufferWorkflow (monitors novel doc accumulation)  │  │
│  │                                                           │  │
│  │  Task Queues:                                             │  │
│  │    assignment, feedback                                   │  │
│  │                                                           │  │
│  │  Workers:                                                 │  │
│  │    2× CPU (pre-warmed embedder + centroid cache)          │  │
│  │                                                           │  │
│  │  SLA: p99 < 500ms for AssignDocumentWorkflow              │  │
│  │  Search attributes:                                       │  │
│  │    account_id (Keyword), assignment_type (Keyword),       │  │
│  │    confidence (Double)                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Why two namespaces

- **Isolation**: Batch workflows generate large histories (many cluster activities). They should not share a visibility store with the real-time path.
- **SLA separation**: Batch workflows tolerate minutes of latency. Assignment workflows need sub-second. Different polling intervals, different worker configs.
- **Independent scaling**: Batch workers can be spot instances (tolerant of interruption — Temporal replays). Assignment workers should be on-demand.

---

## 9. Rate Limiting & Concurrency

### Current monolith approach

```python
# In AsyncLLMClient:
self._semaphore = asyncio.Semaphore(max_concurrency)  # in-process only

# In Stage 4 dispatch:
sem = asyncio.Semaphore(int(os.getenv("CLUSTER_CONCURRENCY", "3")))
```

### Temporal replacement — three layers

**Layer 1: Worker-level (per process)**

```python
Worker(
    ...
    max_concurrent_activities=3,             # replaces CLUSTER_CONCURRENCY semaphore
    max_task_queue_activities_per_second=5,   # throttles LLM call rate per worker
)
```

This is the simplest migration — set the worker config to match current semaphore values.

**Layer 2: Queue-level (global across all workers)**

Temporal Cloud supports `max_task_queue_activities_per_second` as a global rate across all workers on a queue. For self-hosted, use a custom rate limiter:

```python
# Option A: Redis token bucket
@activity.defn
async def process_cluster_with_rate_limit(input):
    async with redis_rate_limiter("azure-openai", tokens_per_second=10):
        return await _do_process_cluster(input)

# Option B: Temporal "semaphore workflow" pattern
# A long-running workflow that manages a counter via signals
# Activities signal "acquire" before LLM calls and "release" after
```

**Layer 3: Per-tenant fairness**

For multi-tenant deployments, use Temporal's **workflow ID prefix** routing:

```python
workflow_id = f"pipeline/{account_id}/{run_id}"
```

Combined with priority task queues (Temporal Cloud feature), this prevents one tenant's 1000-cluster batch from starving another tenant's 5-cluster job.

### Replacing `AsyncLLMClient`

The `AsyncLLMClient` class (semaphore + ThreadPoolExecutor + retry) is replaced by:

- **Concurrency**: Temporal worker's `max_concurrent_activities`
- **Thread pool**: Activity runs in its own thread by default (Temporal Python SDK)
- **Retry**: Temporal activity retry policy (exponential backoff, configurable)
- **Stats tracking**: Temporal metrics (via Prometheus/DataDog interceptors)

The `_sync_call` method (raw Azure OpenAI SDK call) stays as a utility function used inside activities. The wrapper class is no longer needed.

---

## 10. Inter-Service Data Contracts

### Activity Input/Output Schemas

Define these as `@dataclass` types in a shared `models/` package:

```python
# models/pipeline.py

@dataclass
class IngestInput:
    pdf_dir: str
    account_id: str
    max_docs: int | None
    min_chunk_chars: int
    max_chunk_chars: int
    extraction_chunk_chars: int
    max_chunks_per_doc: int
    enable_field_features: bool
    legacy_fields_dir: str | None

@dataclass
class IngestOutput:
    run_id: str
    n_docs: int
    n_embedding_chunks: int
    n_extraction_chunks: int
    n_clauses: int

@dataclass
class EmbedInput:
    run_id: str
    account_id: str
    device: str

@dataclass
class EmbedOutput:
    n_chunk_embeddings: int
    n_summary_embeddings: int

@dataclass
class ClusterInput:
    run_id: str
    account_id: str
    use_optuna: bool
    optuna_trials: int
    evoc_noise_level: float
    guided_keywords: dict[str, list[str]] | None
    guided_alpha: float
    enable_multidim_signals: bool
    field_overlap_weight: float
    obligation_weight: float
    enable_soft_membership: bool
    soft_membership_k: int
    low_confidence_threshold: float

@dataclass
class ClusterOutput:
    n_domains: int
    n_clusters: int
    cluster_ids: list[str]

@dataclass
class ProcessClusterInput:
    cluster_id: str
    run_id: str
    account_id: str
    discovery_mode: str
    max_sample_chunks: int
    max_fields_per_cluster: int
    min_cluster_chunks: int
    discovery_temperature: float
    min_extraction_confidence: float
    extract_values: bool
    enable_confidence_gate: bool
    confidence_gate_threshold: float
    cochran_sample_fraction: float
    cochran_min_sample: int
    cochran_max_sample: int
    personalization_ctx: dict | None

@dataclass
class ProcessClusterOutput:
    cluster_id: str
    fields: int
    extractions: int
    confidence_gate: dict
    profile: dict  # serialized ClusterProfile

@dataclass
class AssignInput:
    pdf_path: str
    account_id: str
    device: str

@dataclass
class AssignOutput:
    filename: str
    agreement_id: str
    cluster_id: str | None
    cluster_label: str
    confidence: float
    assignment_type: str  # "high_confidence" | "tentative" | "novel"
    top_matches: list[tuple[str, float]]
```

### Cross-Workflow Communication

| From | To | Pattern | Mechanism |
|---|---|---|---|
| `AssignDocumentWorkflow` | `NovelBufferWorkflow` | "novel doc added" | Temporal **signal** |
| `NovelBufferWorkflow` | `PipelineWorkflow` | "re-cluster needed" | Temporal **signal** or new workflow start |
| External API | `PipelineWorkflow` | "start pipeline" | Temporal client `start_workflow()` |
| External API | `AssignDocumentWorkflow` | "assign document" | Temporal client `start_workflow()` |
| External API | Feedback service | "submit feedback" | Direct DB write (FeedbackStore) or dedicated activity |

---

## 11. The Real-Time Assignment Path

### Current monolith

`assign_new_documents()` is a standalone async function that:
1. Opens its own DuckDB + embedder connections
2. Loads centroids into `IncrementalClusterAssigner`
3. For each PDF: parse → embed → cosine match → return tier

### Temporal version

```
AssignDocumentWorkflow
  ├─ ParseSinglePDF     (ingestion queue, 5s timeout)
  ├─ EmbedText           (embedding queue, 2s timeout)
  └─ AssignToCluster     (assignment queue, 1s timeout)
      └─ [if novel] → signal NovelBufferWorkflow
```

### Performance consideration

The 500ms SLA for assignment is tight for a 3-activity workflow with Temporal overhead (~20ms per activity dispatch). Options:

1. **Single-activity optimization**: Collapse all 3 steps into one `AssignDocument` activity that runs on a pre-warmed worker with in-memory embedder + centroid cache. This eliminates 2 network hops.

2. **Local activity**: Use Temporal's `executeLocalActivity()` for the embed + assign steps — these run in the workflow worker's process without going through the task queue, eliminating dispatch latency.

3. **Hybrid**: Keep parse as a regular activity (I/O-bound), but embed + assign as local activities on the assignment worker (which has GPU access and cached centroids).

Recommended: **Option 3** (hybrid) — gives you the best latency while keeping parse isolated.

```python
@workflow.defn
class AssignDocumentWorkflow:
    @workflow.run
    async def run(self, input: AssignInput) -> AssignOutput:
        # Regular activity: parsing is I/O-bound, can retry independently
        doc = await workflow.execute_activity(
            parse_single_pdf, input.pdf_path,
            start_to_close_timeout=timedelta(seconds=5),
        )

        # Local activities: run in-process, no task queue dispatch overhead
        embedding = await workflow.execute_local_activity(
            embed_text, doc.text,
            start_to_close_timeout=timedelta(seconds=2),
        )
        result = await workflow.execute_local_activity(
            assign_to_cluster, embedding, input.account_id,
            start_to_close_timeout=timedelta(seconds=1),
        )

        if result.assignment_type == "novel":
            await workflow.signal_external_workflow(
                "novel-buffer", input.account_id,
                "novel_document_added", {"doc_id": doc.agreement_id},
            )

        return result
```

---

## 12. Implementation Phases

### Phase 0: Shared Models + Infrastructure (1 session)

**Goal**: Extract shared types and set up Temporal infrastructure.

```
Tasks:
  □ Create models/ package with all input/output dataclasses
  □ Extract ClusterProfile, MinimalPipelineConfig from minimal_pipeline.py into models/
  □ Set up Temporal dev server (docker-compose or Temporal CLI)
  □ Create worker entry points (gpu_worker.py, llm_worker.py, general_worker.py, assignment_worker.py)
  □ Configure namespaces: extraction-batch, extraction-realtime
  □ Add temporalio SDK to requirements.txt / pyproject.toml
```

### Phase 1: Workflow Shell + Ingestion Activity (1 session)

**Goal**: Get the end-to-end workflow running with Stage 1 as a real activity, stages 2–6 as stubs.

```
Tasks:
  □ Implement PipelineWorkflow with all 7 activity calls (stubs for 2–6)
  □ Implement IngestPDFs activity (extract Stage 1 + Stage 2a from monolith)
  □ Wire up CLI to start PipelineWorkflow via Temporal client
  □ Test: python -m temporal_cli run /path/to/pdfs → workflow visible in Temporal UI
  □ Verify: ingestion activity completes, chunks in DuckDB
```

### Phase 2: Embedding + Clustering Activities (1 session)

**Goal**: Stages 2–3 as real activities on GPU workers.

```
Tasks:
  □ Implement EmbedChunks activity (Stage 2b+2c+2d)
  □ Implement RunClustering activity (full Stage 3)
  □ Extract compute_guided_embeddings, compute_multidimensional_embeddings,
    compute_soft_memberships, extract_cluster_keywords into a shared clustering/ package
  □ Configure GPU worker to poll embedding + clustering queues
  □ Test: full Stages 1→2→3, clusters visible in DuckDB
```

### Phase 3: Extraction Fan-Out (1–2 sessions)

**Goal**: Stage 4 as a child workflow with per-cluster activities.

```
Tasks:
  □ Implement ExtractionFanOutWorkflow (child workflow)
  □ Implement ProcessCluster activity (wraps _process_cluster logic)
  □ Extract _diversity_sample_chunks, _cochran_sample_size, _compute_extraction_confidence,
    _normalize_field, discovery functions into a shared extraction/ package
  □ Move AsyncLLMClient → activities use direct Azure OpenAI calls (retry via Temporal)
  □ Wire personalization: activity fetches PersonalizationContext from FeedbackStore
  □ Test: full Stages 1→4, fields + extractions in DuckDB, profiles as JSON
  □ Test failure isolation: kill one cluster's activity, verify others complete
```

### Phase 4: Intent + Assigner + Manifest Activities (1 session)

**Goal**: Complete the batch pipeline.

```
Tasks:
  □ Implement DiscoverIntents activity (Stage 5, conditional)
  □ Implement InitAssigner activity (Stage 6)
  □ Implement PersistManifest activity (final stats + JSON)
  □ Test: full pipeline end-to-end matches monolith output
  □ Compare: run both monolith and Temporal version on same data, diff DuckDB contents
```

### Phase 5: Real-Time Assignment Workflow (1 session)

**Goal**: `assign` subcommand as a Temporal workflow.

```
Tasks:
  □ Implement AssignDocumentWorkflow in extraction-realtime namespace
  □ Implement ParseSinglePDF, EmbedText, AssignToCluster activities
  □ Implement NovelBufferWorkflow (long-running, signal-driven)
  □ Configure assignment worker with pre-warmed embedder + centroid cache
  □ Benchmark: p99 latency < 500ms for single document assignment
  □ Wire CLI: python -m temporal_cli assign doc.pdf --account-id acct_001
```

### Phase 6: Observability + Hardening (1 session)

**Goal**: Production readiness.

```
Tasks:
  □ Add Temporal search attributes for filtering (account_id, run_id, stage)
  □ Add Prometheus metrics interceptor (activity duration, LLM call count, error rate)
  □ Implement distributed rate limiter for LLM calls (Redis token bucket)
  □ Add dead-letter handling for permanently failed clusters
  □ Set up Temporal UI dashboards for pipeline monitoring
  □ Document runbooks: how to retry a failed workflow, how to cancel, how to replay
```

### Phase 7: Decommission Monolith (1 session)

**Goal**: Remove old code paths.

```
Tasks:
  □ Route CLI through Temporal client instead of direct asyncio.run()
  □ Remove asyncio.Semaphore, AsyncLLMClient, _StageTimer from pipeline
  □ Verify: monolith file is no longer imported by anything except tests
  □ Archive minimal_pipeline.py (or keep as reference)
  □ Update docs/pipeline_design.md to reflect Temporal architecture
```

---

## 13. File Structure (Target)

```
hackathon/
├── temporal/
│   ├── __init__.py
│   ├── client.py                  # Temporal client factory + namespace config
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── pipeline_workflow.py   # PipelineWorkflow (top-level orchestrator)
│   │   ├── extraction_fanout.py   # ExtractionFanOutWorkflow (Stage 4)
│   │   ├── assign_workflow.py     # AssignDocumentWorkflow (real-time)
│   │   └── novel_buffer.py        # NovelBufferWorkflow (signal-driven)
│   ├── activities/
│   │   ├── __init__.py
│   │   ├── ingest.py              # IngestPDFs activity
│   │   ├── embed.py               # EmbedChunks activity
│   │   ├── cluster.py             # RunClustering activity
│   │   ├── extract.py             # ProcessCluster activity
│   │   ├── intent.py              # DiscoverIntents activity
│   │   ├── assign.py              # AssignToCluster + InitAssigner activities
│   │   └── manifest.py            # PersistManifest activity
│   ├── workers/
│   │   ├── gpu_worker.py          # Polls: embedding, clustering
│   │   ├── llm_worker.py          # Polls: discovery-extraction, intent
│   │   ├── general_worker.py      # Polls: ingestion, feedback
│   │   └── assignment_worker.py   # Polls: assignment (pre-warmed)
│   └── models/
│       ├── __init__.py
│       ├── config.py              # MinimalPipelineConfig (extracted)
│       ├── profile.py             # ClusterProfile (extracted)
│       └── activity_io.py         # All Input/Output dataclasses
├── core/
│   ├── minimal_pipeline.py        # Legacy monolith (archived reference)
│   ├── store.py                   # Unchanged — each activity opens its own connection
│   ├── ... (all other core modules unchanged)
```

---

## 14. Testing Strategy

### Unit tests (per activity)

Each activity is testable in isolation by mocking its data store interactions:

```python
async def test_process_cluster_gate_fails():
    """When confidence gate fails, activity should return 0 extractions."""
    input = ProcessClusterInput(
        cluster_id="c_test",
        enable_confidence_gate=True,
        confidence_gate_threshold=0.99,  # impossibly high
        ...
    )
    # Mock DuckDB to return known chunk data
    with mock_store() as store:
        result = await process_cluster(input)
    assert result.extractions == 0
    assert result.confidence_gate["passed"] is False
```

### Integration tests (workflow-level)

Use Temporal's `WorkflowEnvironment` for deterministic testing:

```python
async def test_pipeline_workflow_end_to_end():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        # Register all activities with mock implementations
        worker = Worker(env.client, task_queue="test", workflows=[PipelineWorkflow], activities=[...])
        async with worker:
            result = await env.client.execute_workflow(
                PipelineWorkflow.run,
                PipelineInput(pdf_dir="/test/pdfs", account_id="test"),
                id="test-pipeline",
                task_queue="test",
            )
            assert result.n_clusters > 0
            assert result.n_extractions > 0
```

### Regression tests

Run both monolith and Temporal version on the same PDF corpus, compare:
- Same number of clusters, fields, extractions
- Same ClusterProfile contents (modulo non-deterministic LLM outputs)
- Same DuckDB table row counts

---

## 15. Failure Modes & Mitigations

| Failure | Current Monolith Behavior | Temporal Behavior |
|---|---|---|
| LLM timeout on cluster 7 of 50 | `asyncio.gather(return_exceptions=True)` — logged, other clusters continue | Activity retries 3× with backoff. If permanent, marked failed. Other clusters unaffected. Visible in Temporal UI. |
| GPU OOM during embedding | Pipeline crashes, full restart | EmbedChunks activity fails, retried with potentially smaller batch. Workflow resumes from Stage 2 (Stages 1 output already persisted). |
| HDBSCAN divergence (Optuna) | Pipeline hangs or crashes | RunClustering activity hits `start_to_close_timeout` (15min), retried. Could also add heartbeat-based timeout for Optuna. |
| DuckDB corruption | Pipeline crashes | Activity fails, retried. If persistent, workflow fails with clear error. DuckDB needs external backup strategy. |
| Worker process killed | Everything lost | Temporal replays workflow from last completed activity. No data loss. |
| Deployment during pipeline run | In-flight work lost | Workers gracefully drain. New workers pick up pending activities. Workflow continues seamlessly. |
| 1000-cluster batch blocks other tenants | No mitigation (single process) | Per-tenant workflow IDs + priority queues (Temporal Cloud) or separate task queues per tenant. |

---

## Appendix: Quick Reference — What to Read First

If you're the implementing agent, read these files in order:

1. **This file** — you're here
2. **`core/minimal_pipeline.py` top comment** (sections 1–12) — complete architectural breakdown with function index, config inventory, data-store map, module groupings
3. **`core/minimal_pipeline.py` Stage 3 block** (~lines 1770–1960) — the clustering logic is the most complex stage to extract
4. **`core/minimal_pipeline.py` `_process_cluster()`** (~lines 978–1195) — the per-cluster 3-phase logic that becomes the `ProcessCluster` activity
5. **`core/minimal_pipeline.py` `assign_new_documents()`** (~lines 2210–2380) — the real-time path that becomes its own workflow
6. **`core/store.py`** — understand the DuckDB schema before splitting data access across activities
7. **`core/feedback_store.py` + `core/personalization.py`** — the cross-cutting concern that touches multiple services
