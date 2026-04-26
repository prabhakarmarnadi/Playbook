# Temporal Microservices Design

**Date:** April 23, 2026  
**Status:** Proposed implementation design  
**Primary source of truth:** `core/minimal_pipeline.py`  
**Companion docs:** `docs/FRONTEND_API_SPEC.md`, `docs/TEMPORAL_DECOMPOSITION_PLAN.md`, `docs/pipeline_design_v2.md`

---

## 1. Why this doc exists

The current system has two mismatches:

1. The backend implementation is much richer than the current API surface. The existing API only exposes a small job launcher (`/cluster`) plus job status, while the product already supports document ingestion, ontology exploration, extraction review, search, risk analysis, feedback, personalization, virtual clusters, and incremental assignment.
2. The current pipeline is a monolith. It is difficult to scale ingestion, GPU embedding, clustering, LLM-heavy discovery, extraction, and low-latency assignment independently.

This document focuses on how to break the monolith into Temporal-orchestrated services inside a single repository.

The frontend-facing API material has been extracted to `docs/FRONTEND_API_SPEC.md` so product-facing API work and backend orchestration work can evolve independently.

Assumption: the backend is a queued batch system. The frontend/API implications of that model are documented in `docs/FRONTEND_API_SPEC.md`.

---

## 2. Frontend API companion

The frontend-facing API contract, feedback taxonomy, and eventing model have been extracted to `docs/FRONTEND_API_SPEC.md`.

Use that document for:

1. product-facing API route design
2. frontend integration work
3. feedback and review queue API design
4. run-progress and event streaming contracts

This document now focuses on Temporal orchestration, worker boundaries, shared workflow contracts, and implementation handoff.

---

## 3. Proposed Temporal-based microservice architecture

The frontend-facing API above should sit on top of Temporal workflows and a small set of independently scalable worker services.

### 3.1 Service boundaries

| Service | Responsibility | Owns Temporal roles? | Main dependencies |
|---|---|---|---|
| `api-gateway` | REST, auth, validation, SSE, BFF aggregation | starts workflows, sends signals, reads queries | FastAPI, auth, query clients |
| `orchestrator` | Workflow definitions and command translation | yes | Temporal client, shared contracts |
| `ingestion-worker` | parse PDFs, validate batches, chunk generation, field features | activity worker | `pdf_parser`, `chunker`, `field_features` |
| `embedding-worker` | Nomic embedding, summary embedding, LanceDB writes | activity worker | `nomic_embedder`, `macro_clusterer`, `lancedb_store` |
| `clustering-worker` | macro/micro clustering, guided clustering, multidim embeddings, soft membership | activity worker | `evoc_pipeline`, `topic_merger` |
| `discovery-worker` | RLM V3 discovery, schema normalization, profile build prep | activity worker | `rlm_v3`, shared models |
| `extraction-worker` | confidence gate, RAG extraction, intent persistence, profile assembly | activity worker | `rag_extractor`, `intent_extractor`, `store` |
| `assignment-worker` | incremental assignment, tentative and novel routing | activity worker | `cluster_assigner`, `pdf_parser`, `nomic_embedder` |
| `feedback-worker` | feedback writes, auto-tune, virtual cluster promotion, overrides | activity worker | `feedback_store`, `personalization` |
| `query-service` | read-model endpoints for dashboards and search | no | `store`, `lancedb_store`, search/KG/risk modules |

### 3.2 Temporal workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `PipelineRunWorkflow` | `POST /runs` or batch start | full pipeline orchestration |
| `ClusterProcessingWorkflow` | child workflow from pipeline | fan-out per cluster for discovery/extraction |
| `DocumentAssignmentWorkflow` | `POST /assignments` | real-time or near-real-time assignment |
| `DocumentReprocessWorkflow` | document reprocess request | rerun selected stages on subset |
| `FeedbackProjectionWorkflow` | feedback writes | recompute personalization read models |
| `EvaluationWorkflow` | analytics request | run evaluation bundle |
| `RiskScoringWorkflow` | risk request | heuristic or LLM scoring |

### 3.3 Activity mapping from the monolith

| Monolith function or block | New activity |
|---|---|
| Stage 1 parse block in `run_minimal_pipeline` | `ParseBatchDocumentsActivity` |
| Stage 2 chunk and field-feature block | `GenerateChunksActivity` |
| Stage 2 embedding and persistence block | `EmbedAndPersistActivity` |
| Stage 3 clustering block | `ClusterCorpusActivity` |
| `_process_cluster` discovery portion | `DiscoverClusterSchemaActivity` |
| `_process_cluster` confidence gate | `RunConfidenceGateActivity` |
| `_process_cluster` extraction portion | `ExtractClusterValuesActivity` |
| `_build_cluster_profile` | `BuildClusterProfileActivity` |
| Stage 5 assigner initialization | `RefreshAssignerCacheActivity` |
| `assign_new_documents` | `AssignDocumentsActivity` |
| feedback store updates | `ApplyFeedbackEventActivity` |
| personalization auto-tune | `AutoTunePersonalizationActivity` |

### 3.4 Queue topology

| Temporal task queue | Worker | Notes |
|---|---|---|
| `pipeline.ingestion` | ingestion-worker | CPU + I/O |
| `pipeline.embedding` | embedding-worker | GPU-bound |
| `pipeline.clustering` | clustering-worker | CPU/GPU depending mode |
| `pipeline.discovery` | discovery-worker | LLM-bound |
| `pipeline.extraction` | extraction-worker | LLM + retrieval bound |
| `pipeline.assignment` | assignment-worker | low-latency |
| `pipeline.feedback` | feedback-worker | low-latency |
| `pipeline.analytics` | query-service or dedicated analytics worker | optional |

### 3.5 High-level workflow execution

```text
Frontend -> api-gateway -> PipelineRunWorkflow

PipelineRunWorkflow
  -> ParseBatchDocumentsActivity
  -> GenerateChunksActivity
  -> EmbedAndPersistActivity
  -> ClusterCorpusActivity
  -> N x ClusterProcessingWorkflow
       -> DiscoverClusterSchemaActivity
       -> RunConfidenceGateActivity
       -> ExtractClusterValuesActivity
       -> BuildClusterProfileActivity
  -> RefreshAssignerCacheActivity
  -> PersistRunManifestActivity
```

### 3.6 Why split discovery and extraction into separate services

The current monolith combines them inside `_process_cluster`, but the operating profile is different.

1. Discovery is schema-generation heavy and should be retried at cluster granularity.
2. Extraction is volume-heavy and may need separate throttling.
3. Confidence gating sits naturally between them.
4. The frontend needs to surface partial success: discovery succeeded but extraction skipped or failed.

### 3.7 Distinguished engineer recommendation

If this system were being decomposed at a top ML firm, the recommendation would be:

1. Do not split by helper function.
2. Do not split by every stage mechanically.
3. Split by resource profile, synchronization boundary, retry boundary, and latency requirement.

From this file, there are really five execution regimes:

1. Document-parallel CPU and I/O work: parse, cache, chunk, field features.
2. Corpus-global GPU and clustering work: embeddings, summary embeddings, macro and micro clustering.
3. Cluster-parallel LLM work: discovery, confidence gating, extraction, profile build.
4. Low-latency online routing: incremental assignment.
5. Tenant-scoped control-plane logic: feedback and personalization.

That is why the recommended production cut is:

1. `api-gateway`
2. `ingestion-service`
3. `embedding-service`
4. `clustering-service`
5. `discovery-service`
6. `extraction-service`
7. `assignment-service`
8. `feedback-personalization-service`
9. `query-service`

This is the correct level of granularity. Anything finer creates orchestration noise, unstable contracts, and payload churn. Anything coarser forces GPU work, LLM work, and online routing to contend for the same queues and failure domains.

### 3.8 The actual seams in `minimal_pipeline.py`

The code already exposes the right service boundaries.

| Current code surface | Recommended service boundary |
|---|---|
| `run_minimal_pipeline()` | top-level Temporal workflow |
| Stage 1 parse block | ingestion service |
| Stage 2 chunk generation and field features | ingestion service |
| Stage 2 embedding and LanceDB persistence | embedding service |
| Stage 3 macro and micro clustering | clustering service |
| `_process_cluster()` | child workflow boundary |
| `_discover_fields_standard()`, `_discover_fields_rlm_v3()`, `_discover_and_extract_rlm_v3()` | discovery service |
| `_run_rag_extraction()` | extraction service |
| `_build_cluster_profile()` | extraction or profile assembly activity |
| `assign_new_documents()` | assignment service |
| feedback and personalization initialization | feedback-personalization service |

The most important boundary is `_process_cluster()`. That is the natural Temporal child-workflow unit because it matches the current concurrency model, the current cost model, and the desired retry isolation.

### 3.9 What Temporal should and should not orchestrate

Temporal should orchestrate the batch and control-plane flows. It should not become the transport mechanism for large data movement, and it should not sit directly on the synchronous hot path if the assignment SLA matters.

Temporal should orchestrate:

1. Full pipeline runs.
2. Per-cluster discovery and extraction fan-out.
3. Document reprocessing.
4. Feedback-triggered recomputations.
5. Risk scoring and evaluation jobs.

Temporal should not be the primary carrier for:

1. Large embedding matrices.
2. Large chunk payloads.
3. Full document text blobs.
4. The inline, low-latency request path for single-document assignment.

The assignment service can still use Temporal for side effects such as audit persistence, novel-buffer signals, or deferred centroid refresh. The actual assignment decision should remain a direct service call backed by a prewarmed embedder and in-memory centroid cache.

### 3.10 Hard architectural rules

These rules should be treated as design constraints, not optional preferences.

1. Workflow payloads must contain references, IDs, or artifact paths, not full embeddings or large chunk arrays.
2. No GPU or LLM work belongs in workflow code; all of it belongs in activities.
3. Stage 3 clustering remains a corpus-global barrier and should not be decomposed into per-document clustering tasks.
4. Long-running activities must heartbeat and support cancellation.
5. LLM concurrency must be controlled by distributed rate limiting, not just Temporal queue concurrency.
6. The assignment path must be isolated from batch extraction capacity.
7. The first migration should keep DuckDB and LanceDB as the source of truth rather than combining orchestration migration with a database re-platform.
8. Idempotency must move from UUID-only generation to deterministic request or content-based keys where feasible.

### 3.11 Recommended first migration shape

The safest path is not a rewrite. It is a Temporal wrapper around the current implementation seams.

1. Extract shared contracts and config models from the monolith.
2. Wrap current stage logic in activity adapters.
3. Stand up `PipelineRunWorkflow` and per-cluster child workflows.
4. Move the frontend to the new API gateway and run-status APIs.
5. Split out the assignment service as a dedicated online path.
6. Add explicit read models for dashboards, review queues, and cluster profile queries.

This preserves the validated algorithmic core while changing only orchestration, scaling boundaries, and API surface in the first pass.

---

## 4. Single-repo structure

The implementation should remain in one repository to keep contracts, workflow code, data access, and UI-adjacent API logic versioned together.

### 4.1 Proposed layout

```text
hackathon/
  services/
    api_gateway/
      app/
        routes/
        schemas/
        sse/
        auth/
    orchestrator/
      workflows/
      signals/
      queries/
    query_service/
      readers/
      projections/
  workers/
    ingestion/
    embedding/
    clustering/
    discovery/
    extraction/
    assignment/
    feedback/
  packages/
    contracts/
      api/
      events/
      workflows/
    domain/
      models/
      config/
    persistence/
      duckdb/
      lancedb/
    pipeline/
      adapters/
      compatibility/
  core/
    ...existing modules...
  docs/
    ...design docs...
```

### 4.2 Shared packages required

| Package | Purpose |
|---|---|
| `packages/contracts/api` | Pydantic request and response models |
| `packages/contracts/workflows` | Temporal workflow payloads and result models |
| `packages/domain/models` | `ClusterProfile`, config models, review item models |
| `packages/persistence` | thin wrappers around current DuckDB and LanceDB adapters |
| `packages/pipeline/adapters` | compatibility layer that reuses existing `core/*` implementations |

---

## 5. Read model strategy

The frontend should not reconstruct pages by stitching raw tables client-side. Build explicit read models in the query service.

### 5.1 Required read models

1. `run_summary_view`
2. `run_stage_progress_view`
3. `document_detail_view`
4. `cluster_detail_view`
5. `cluster_profile_view`
6. `ontology_tree_view`
7. `review_queue_view`
8. `feedback_summary_view`
9. `assignment_summary_view`
10. `dashboard_metrics_view`

### 5.2 Near-term implementation

To reduce migration risk, these read models can initially be assembled from current stores and JSON artifacts:

1. DuckDB main store
2. DuckDB feedback sidecar
3. LanceDB vector store
4. Artifact files under `data/artifacts/<account>/<run_id>/`

Later, if query latency becomes a bottleneck, materialize selected views into Postgres or another operational query store.

---

## 6. API-to-service mapping

| External API area | Primary service | Temporal involvement |
|---|---|---|
| batches, runs | api-gateway + orchestrator | start workflow, signal, query history |
| documents, clusters, ontology | query-service | no workflow start |
| field and extraction review | api-gateway + extraction-worker + feedback-worker | commands become workflow signals or activities |
| feedback and personalization | api-gateway + feedback-worker | async for write, query-service for read |
| incremental assignment | api-gateway + assignment-worker | one short workflow or direct activity wrapper |
| analytics and risk | api-gateway + analytics worker | queued workflows |

---

## 7. Implementation phases

### Phase 1: Stabilize contracts

1. Extract shared Pydantic models for runs, clusters, fields, extractions, feedback events, and assignments.
2. Build the new REST surface in the API gateway, even if some endpoints still call monolith helpers directly.
3. Add SSE run progress endpoint.

### Phase 2: Temporalize the pipeline core

1. Implement `PipelineRunWorkflow` plus ingestion, embedding, clustering, discovery, and extraction activities.
2. Keep DuckDB and LanceDB storage unchanged.
3. Reuse existing `core/*` logic through thin adapters rather than rewriting algorithms.

### Phase 3: Temporalize feedback and assignment

1. Implement `DocumentAssignmentWorkflow`.
2. Implement feedback commands as workflows or activities.
3. Add review queue projections.

### Phase 4: Query-service hardening

1. Add dedicated read models for the dashboard, ontology tree, and cluster detail views.
2. Add caching for expensive cluster map, graph, and risk queries.
3. Add export endpoints.

### Phase 5: Remove monolithic orchestration

1. Leave `core/minimal_pipeline.py` as a compatibility layer for CLI and offline experiments.
2. Move production execution to the workflow-backed services.
3. Deprecate the current thin `api_server.py` routes after parity is reached.

---

## 8. Immediate implementation decisions

These decisions reduce ambiguity for the first implementation pass.

1. The frontend will call one API gateway, not individual workers.
2. Temporal is internal only; workflow IDs are exposed, but Temporal APIs are not.
3. All long-running commands return a job handle plus SSE stream link.
4. The API will expose both specialized feedback endpoints and a generic `/feedback/events` endpoint.
5. The current DuckDB and LanceDB stores remain the source of truth in the first migration.
6. `ClusterProfile` JSON artifacts remain valid, but the query service will expose them as API resources.
7. Tentative assignments and novel documents become first-class review queues.

---

## 9. Recommended first backlog

The first engineering slice should deliver end-to-end value without rewriting the whole platform.

1. Add a new API gateway module with routes for batches, runs, documents, clusters, fields, feedback, and assignments.
2. Extract shared request and response schemas from current API and UI needs.
3. Implement `PipelineRunWorkflow` using the current monolith internals through adapter activities.
4. Add SSE progress events sourced from workflow state.
5. Add specialized feedback endpoints for cluster correction, extraction correction, intent override, virtual cluster creation, threshold update, and auto-tune.
6. Add query endpoints for cluster profile, ontology tree, review queues, and assignment buckets.

---

## 10. Success criteria

This design is successful when:

1. Every current Streamlit workflow has a stable API equivalent.
2. Drag-and-drop batch ingestion, review, and incremental assignment work without direct database access.
3. Discovery and extraction can scale independently.
4. The frontend can inspect run progress in real time.
5. Feedback and personalization are explicit API concepts rather than hidden side effects.
6. The system can evolve from a monolith to Temporal workers without changing the frontend contract.

---

## 11. Concrete Temporal workflow and activity spec

This section turns the architecture into an implementation contract. The goal is to make it possible to start building Temporal workflows and workers without reopening the decomposition question.

### 11.1 Contract design rules

1. Workflow payloads must be small, stable, and versioned.
2. Activities must take references to persisted data, not raw corpus-scale arrays.
3. All GPU, LLM, and database work stays inside activities.
4. All workflow outputs must be queryable by the API gateway without reading Temporal history directly.
5. Every workflow and activity contract in this section is intended to live in `packages/contracts/workflows`.

### 11.2 Common scalar types

Use these common aliases across workflow payloads:

| Type | Shape | Notes |
|---|---|---|
| `RunId` | `str` | e.g. `run_ab12cd34` |
| `AccountId` | `str` | storage and partition key |
| `TenantId` | `str | None` | personalization partition key |
| `WorkflowRef` | `str` | Temporal workflow ID |
| `ArtifactRef` | `str` | artifact path, DB key, or object-store URI |
| `ChunkId` | `str` | stable chunk identifier |
| `ClusterId` | `str` | stable cluster identifier |
| `AgreementId` | `str` | stable document/agreement identifier |
| `IsoTimestamp` | `str` | ISO-8601 UTC string |

### 11.3 Canonical payload models

These are the minimum shared models that should exist before worker implementation starts.

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StorageRefs:
  db_path: str
  lance_path: str
  artifacts_root: str
  parsed_text_cache_dir: str | None = None


@dataclass
class RunContext:
  run_id: str
  account_id: str
  tenant_id: str | None = None
  workflow_id: str = ""
  submitted_at: str = ""
  storage: StorageRefs | None = None


@dataclass
class ActivityExecutionMeta:
  attempt: int = 1
  requested_by: str | None = None
  priority: str = "normal"
  tags: dict[str, str] = field(default_factory=dict)


@dataclass
class PersonalizationContextPayload:
  tenant_id: str = ""
  high_confidence_threshold: float = 0.85
  tentative_threshold: float = 0.60
  outlier_min_similarity: float = 0.35
  intent_match_threshold: float = 0.85
  intent_temperature: float = 0.0
  cluster_weights: dict[str, float] = field(default_factory=dict)
  edge_weights: dict[str, float] = field(default_factory=dict)
  virtual_clusters: list[dict[str, Any]] = field(default_factory=list)
  intent_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
  correction_examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProcessingStats:
  status: str = "queued"
  started_at: str | None = None
  completed_at: str | None = None
  elapsed_s: float | None = None
  warnings: list[str] = field(default_factory=list)
  errors: list[str] = field(default_factory=list)
```

### 11.4 Workflow ID and idempotency scheme

Use deterministic workflow IDs so a retried API request does not create duplicate runs.

| Workflow | Workflow ID format |
|---|---|
| `PipelineRunWorkflow` | `pipeline/{account_id}/{run_id}` |
| `ClusterProcessingWorkflow` | `cluster/{account_id}/{run_id}/{cluster_id}` |
| `DocumentAssignmentWorkflow` | `assign/{account_id}/{request_id}` |
| `DocumentReprocessWorkflow` | `reprocess/{account_id}/{document_id}/{request_id}` |
| `FeedbackProjectionWorkflow` | `feedback/{tenant_id}/{event_id}` |
| `EvaluationWorkflow` | `evaluation/{account_id}/{evaluation_id}` |
| `RiskScoringWorkflow` | `risk/{account_id}/{request_id}` |

API requests that create workflows should also require an `Idempotency-Key` header and persist the mapping from key to workflow ID.

### 11.5 Top-level workflow contracts

#### 11.5.1 `PipelineRunWorkflow`

This is the production replacement for `run_minimal_pipeline()`.

```python
@dataclass
class PipelineRunWorkflowInput:
  run_context: RunContext
  pdf_source_ref: str
  source_kind: str  # batch_id | filesystem_dir | blob_prefix
  config: dict[str, Any]
  resume_from_stage: int | None = None
  max_clusters: int | None = None
  notify_on_complete: bool = True
  notify_on_failure: bool = True


@dataclass
class PipelineRunWorkflowOutput:
  run_id: str
  account_id: str
  status: str  # completed | failed | canceled | partial_success
  stage_statuses: list[dict[str, Any]]
  n_documents: int = 0
  n_chunks: int = 0
  n_domains: int = 0
  n_clusters: int = 0
  n_fields: int = 0
  n_extractions: int = 0
  n_rlm_intents: int = 0
  profiles_dir: str | None = None
  manifest_path: str | None = None
  llm_stats: dict[str, Any] = field(default_factory=dict)
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Execution order:

1. `ResolvePersonalizationContextActivity`
2. `ParseBatchDocumentsActivity`
3. `GenerateChunksActivity`
4. `EmbedCorpusActivity`
5. `ClusterCorpusActivity`
6. `ClusterFanoutWorkflow`
7. `RefreshAssignmentCacheActivity`
8. `PersistRunManifestActivity`

#### 11.5.2 `ClusterFanoutWorkflow`

This is an internal workflow used by `PipelineRunWorkflow` to fan out Stage 4 work.

```python
@dataclass
class ClusterFanoutWorkflowInput:
  run_context: RunContext
  cluster_ids: list[str]
  config: dict[str, Any]
  personalization_ctx: PersonalizationContextPayload | None = None


@dataclass
class ClusterFanoutWorkflowOutput:
  total_clusters: int
  succeeded_clusters: int
  failed_clusters: int
  skipped_clusters: int
  total_fields: int
  total_extractions: int
  total_rlm_intents: int
  profile_refs: list[str] = field(default_factory=list)
  per_cluster_results: list[dict[str, Any]] = field(default_factory=list)
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Child workflow behavior:

1. Start one `ClusterProcessingWorkflow` per cluster.
2. Let worker queue concurrency bound active cluster work.
3. Aggregate per-cluster outcomes into a single summary for the parent workflow.

#### 11.5.3 `ClusterProcessingWorkflow`

This is the natural Temporal replacement for `_process_cluster()` as an orchestration boundary.

```python
@dataclass
class ClusterProcessingWorkflowInput:
  run_context: RunContext
  cluster_id: str
  config: dict[str, Any]
  personalization_ctx: PersonalizationContextPayload | None = None


@dataclass
class ClusterProcessingWorkflowOutput:
  cluster_id: str
  cluster_label: str
  discovery_status: str  # completed | failed | fallback_used
  extraction_status: str  # completed | skipped_by_gate | failed | no_extraction
  field_count: int = 0
  extraction_count: int = 0
  rlm_intent_count: int = 0
  confidence_gate: dict[str, Any] = field(default_factory=dict)
  profile_ref: str | None = None
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Execution order:

1. `LoadClusterProcessingContextActivity`
2. `DiscoverClusterSchemaActivity`
3. `PersistDiscoveredFieldsActivity`
4. `PersistRlmIntentsActivity`
5. `RunConfidenceGateActivity`
6. `ExtractClusterValuesActivity` if gate passed and extraction enabled
7. `BuildClusterProfileActivity`

#### 11.5.4 `DocumentAssignmentWorkflow`

This workflow exists for asynchronous side effects and auditability. The synchronous assignment decision should still come from the assignment service directly.

```python
@dataclass
class DocumentAssignmentWorkflowInput:
  run_context: RunContext
  request_id: str
  document_ref: str
  document_source_kind: str  # upload | blob | filesystem
  config: dict[str, Any]


@dataclass
class DocumentAssignmentWorkflowOutput:
  request_id: str
  document_id: str
  cluster_id: str | None
  cluster_label: str | None
  confidence: float
  assignment_type: str  # high_confidence | tentative | novel
  top_matches: list[dict[str, Any]] = field(default_factory=list)
  persisted_novel_document: bool = False
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

#### 11.5.5 `DocumentReprocessWorkflow`

```python
@dataclass
class DocumentReprocessWorkflowInput:
  run_context: RunContext
  document_id: str
  stages: list[str]  # parse | chunk | embed | assign | extract
  reason: str = ""
  config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentReprocessWorkflowOutput:
  document_id: str
  stages_run: list[str] = field(default_factory=list)
  status: str = "completed"
  result_refs: dict[str, str] = field(default_factory=dict)
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

#### 11.5.6 `FeedbackProjectionWorkflow`

```python
@dataclass
class FeedbackProjectionWorkflowInput:
  tenant_id: str
  event_id: str
  entity_type: str
  entity_id: str
  action: str
  payload: dict[str, Any]


@dataclass
class FeedbackProjectionWorkflowOutput:
  tenant_id: str
  event_id: str
  status: str
  updated_projection_keys: list[str] = field(default_factory=list)
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

### 11.6 Activity contracts

The activity list below is intentionally concrete and should map one-to-one to worker packages.

#### 11.6.1 `ResolvePersonalizationContextActivity`

Task queue: `pipeline.feedback`

```python
@dataclass
class ResolvePersonalizationContextInput:
  run_context: RunContext


@dataclass
class ResolvePersonalizationContextOutput:
  personalization_ctx: PersonalizationContextPayload | None
  feedback_enabled: bool
  feedback_summary: dict[str, Any] = field(default_factory=dict)
```

Behavior:

1. Open `FeedbackStore` for `tenant_id or account_id`.
2. Build `PersonalizationContext` via `PersonalizationEngine.get_context()`.
3. Return a serialized payload usable by downstream workflows and activities.

#### 11.6.2 `ParseBatchDocumentsActivity`

Task queue: `pipeline.ingestion`

```python
@dataclass
class ParseBatchDocumentsInput:
  run_context: RunContext
  pdf_source_ref: str
  source_kind: str
  max_docs: int | None = None
  use_cache: bool = True


@dataclass
class ParseBatchDocumentsOutput:
  run_id: str
  document_count: int
  agreement_ids: list[str] = field(default_factory=list)
  parsed_documents_ref: str = ""
  cache_hit: bool = False
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Implementation anchor:

1. `core.pdf_parser.parse_pdf_directory`
2. `core.pdf_parser.load_parsed_agreements`

#### 11.6.3 `GenerateChunksActivity`

Task queue: `pipeline.ingestion`

```python
@dataclass
class GenerateChunksInput:
  run_context: RunContext
  parsed_documents_ref: str
  config: dict[str, Any]


@dataclass
class GenerateChunksOutput:
  run_id: str
  document_count: int
  chunk_count: int
  clause_count: int
  chunk_manifest_ref: str
  clause_manifest_ref: str
  field_feature_summary: dict[str, Any] = field(default_factory=dict)
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Implementation anchor:

1. `core.chunker.batch_generate_unified_chunks`
2. `core.field_features.FieldFeatureExtractor`

#### 11.6.4 `EmbedCorpusActivity`

Task queue: `pipeline.embedding`

```python
@dataclass
class EmbedCorpusInput:
  run_context: RunContext
  chunk_manifest_ref: str
  config: dict[str, Any]


@dataclass
class EmbedCorpusOutput:
  run_id: str
  chunk_count: int
  summary_count: int
  chunk_embeddings_ref: str
  summary_embeddings_ref: str
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Implementation anchor:

1. `core.nomic_embedder.NomicEmbedder.get_instance`
2. `core.macro_clusterer.get_field_summaries`
3. `core.lancedb_store.LanceVectorStore.upsert_chunks`
4. `core.lancedb_store.LanceVectorStore.upsert_documents`

#### 11.6.5 `ClusterCorpusActivity`

Task queue: `pipeline.clustering`

```python
@dataclass
class ClusterCorpusInput:
  run_context: RunContext
  chunk_embeddings_ref: str
  summary_embeddings_ref: str
  chunk_manifest_ref: str
  config: dict[str, Any]


@dataclass
class ClusterCorpusOutput:
  run_id: str
  domain_count: int
  cluster_count: int
  cluster_ids: list[str] = field(default_factory=list)
  centroids_ref: str = ""
  soft_memberships_ref: str | None = None
  low_confidence_chunks_ref: str | None = None
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Implementation anchor:

1. `core.evoc_pipeline.evoc_optimize`, `evoc_cluster`, `_select_best_layer`, `_llm_label_clusters`
2. `compute_guided_embeddings`
3. `compute_multidimensional_embeddings`
4. `compute_soft_memberships`
5. `identify_low_confidence_chunks`
6. `extract_cluster_keywords`

#### 11.6.6 `LoadClusterProcessingContextActivity`

Task queue: `pipeline.discovery`

```python
@dataclass
class LoadClusterProcessingContextInput:
  run_context: RunContext
  cluster_id: str


@dataclass
class LoadClusterProcessingContextOutput:
  cluster_id: str
  cluster_label: str
  cluster_keywords: list[str] = field(default_factory=list)
  chunk_rows_ref: str = ""
  chunk_count: int = 0
  agreement_count: int = 0
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Purpose:

1. Load cluster metadata.
2. Resolve member chunk rows.
3. Externalize large chunk payloads behind a reference so downstream activities do not duplicate DB reads unnecessarily.

#### 11.6.7 `DiscoverClusterSchemaActivity`

Task queue: `pipeline.discovery`

```python
@dataclass
class DiscoverClusterSchemaInput:
  run_context: RunContext
  cluster_id: str
  chunk_rows_ref: str
  config: dict[str, Any]
  personalization_ctx: PersonalizationContextPayload | None = None


@dataclass
class DiscoverClusterSchemaOutput:
  cluster_id: str
  cluster_label: str
  fields: list[dict[str, Any]] = field(default_factory=list)
  rlm_meta: dict[str, Any] = field(default_factory=dict)
  inline_extractions: list[dict[str, Any]] = field(default_factory=list)
  discovery_strategy: str = ""
  used_fallback: bool = False
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Implementation anchor:

1. `_diversity_sample_chunks`
2. `_discover_and_extract_rlm_v3`
3. `_discover_fields_rlm_v3`
4. `_discover_fields_standard`
5. `_normalize_field`

#### 11.6.8 `PersistDiscoveredFieldsActivity`

Task queue: `pipeline.discovery`

```python
@dataclass
class PersistDiscoveredFieldsInput:
  run_context: RunContext
  cluster_id: str
  fields: list[dict[str, Any]]


@dataclass
class PersistDiscoveredFieldsOutput:
  cluster_id: str
  field_count: int
  field_id_map: dict[str, str] = field(default_factory=dict)
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

#### 11.6.9 `PersistRlmIntentsActivity`

Task queue: `pipeline.discovery`

```python
@dataclass
class PersistRlmIntentsInput:
  run_context: RunContext
  cluster_id: str
  rlm_meta: dict[str, Any]


@dataclass
class PersistRlmIntentsOutput:
  cluster_id: str
  intent_count: int
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Implementation anchor:

1. `core.intent_extractor.IntentRegistry`
2. `core.store.ClusteringStore.insert_clause_intent`

#### 11.6.10 `RunConfidenceGateActivity`

Task queue: `pipeline.extraction`

```python
@dataclass
class RunConfidenceGateInput:
  run_context: RunContext
  cluster_id: str
  cluster_label: str
  chunk_rows_ref: str
  fields: list[dict[str, Any]]
  config: dict[str, Any]
  rlm_meta: dict[str, Any] = field(default_factory=dict)
  personalization_ctx: PersonalizationContextPayload | None = None


@dataclass
class RunConfidenceGateOutput:
  cluster_id: str
  enabled: bool
  passed: bool
  metrics: dict[str, Any] = field(default_factory=dict)
  sampled_chunk_ids: list[str] = field(default_factory=list)
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Implementation anchor:

1. `_cochran_sample_size`
2. `_compute_extraction_confidence`
3. `_run_rag_extraction`

#### 11.6.11 `ExtractClusterValuesActivity`

Task queue: `pipeline.extraction`

```python
@dataclass
class ExtractClusterValuesInput:
  run_context: RunContext
  cluster_id: str
  cluster_label: str
  fields: list[dict[str, Any]]
  field_id_map: dict[str, str]
  chunk_rows_ref: str
  config: dict[str, Any]
  rlm_meta: dict[str, Any] = field(default_factory=dict)
  inline_extractions: list[dict[str, Any]] = field(default_factory=list)
  personalization_ctx: PersonalizationContextPayload | None = None


@dataclass
class ExtractClusterValuesOutput:
  cluster_id: str
  extraction_count: int
  rag_meta: dict[str, Any] = field(default_factory=dict)
  extraction_refs: list[str] = field(default_factory=list)
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Implementation anchor:

1. `_build_rlm_context_section`
2. `_run_rag_extraction`
3. `core.rag_extractor.RAGExtractor.extract_cluster`

#### 11.6.12 `BuildClusterProfileActivity`

Task queue: `pipeline.extraction`

```python
@dataclass
class BuildClusterProfileInput:
  run_context: RunContext
  cluster_id: str
  fields: list[dict[str, Any]]
  rlm_meta: dict[str, Any]
  extraction_summary: dict[str, Any]
  confidence_gate: dict[str, Any]
  soft_membership_ref: str | None = None


@dataclass
class BuildClusterProfileOutput:
  cluster_id: str
  profile_ref: str
  profile_summary: dict[str, Any] = field(default_factory=dict)
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Implementation anchor:

1. `_build_cluster_profile`
2. `ClusterProfile.to_dict`

#### 11.6.13 `RefreshAssignmentCacheActivity`

Task queue: `pipeline.assignment`

```python
@dataclass
class RefreshAssignmentCacheInput:
  run_context: RunContext
  personalization_ctx: PersonalizationContextPayload | None = None


@dataclass
class RefreshAssignmentCacheOutput:
  account_id: str
  centroid_count: int
  high_threshold: float
  tentative_threshold: float
  cache_generation: str
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

#### 11.6.14 `AssignDocumentActivity`

Task queue: `pipeline.assignment`

```python
@dataclass
class AssignDocumentInput:
  run_context: RunContext
  document_ref: str
  config: dict[str, Any]
  personalization_ctx: PersonalizationContextPayload | None = None


@dataclass
class AssignDocumentOutput:
  document_id: str
  cluster_id: str | None
  cluster_label: str | None
  confidence: float
  assignment_type: str
  top_matches: list[dict[str, Any]] = field(default_factory=list)
  persisted_novel_document: bool = False
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

Implementation anchor:

1. `assign_new_documents`
2. `core.cluster_assigner.IncrementalClusterAssigner`

#### 11.6.15 `PersistRunManifestActivity`

Task queue: `pipeline.ingestion`

```python
@dataclass
class PersistRunManifestInput:
  run_context: RunContext
  pipeline_output: PipelineRunWorkflowOutput


@dataclass
class PersistRunManifestOutput:
  run_id: str
  manifest_path: str
  stats: ProcessingStats = field(default_factory=ProcessingStats)
```

### 11.7 Task queue ownership

| Task queue | Activities owned |
|---|---|
| `pipeline.ingestion` | `ParseBatchDocumentsActivity`, `GenerateChunksActivity`, `PersistRunManifestActivity` |
| `pipeline.embedding` | `EmbedCorpusActivity` |
| `pipeline.clustering` | `ClusterCorpusActivity` |
| `pipeline.discovery` | `LoadClusterProcessingContextActivity`, `DiscoverClusterSchemaActivity`, `PersistDiscoveredFieldsActivity`, `PersistRlmIntentsActivity` |
| `pipeline.extraction` | `RunConfidenceGateActivity`, `ExtractClusterValuesActivity`, `BuildClusterProfileActivity` |
| `pipeline.assignment` | `RefreshAssignmentCacheActivity`, `AssignDocumentActivity` |
| `pipeline.feedback` | `ResolvePersonalizationContextActivity`, feedback projection activities |

### 11.8 Default retry and timeout policies

These values are the starting point and should be encoded in workflow helpers rather than copied per workflow.

| Activity class | Start-to-close | Heartbeat | Retry policy |
|---|---|---|---|
| parsing and chunking | 10m | 30s | 2 attempts, 10s initial backoff |
| embedding | 20m | 30s | 2 attempts, 20s initial backoff |
| clustering | 30m | 60s | 2 attempts, 30s initial backoff |
| discovery | 15m | 30s | 3 attempts, 10s initial backoff |
| extraction | 20m | 30s | 3 attempts, 10s initial backoff |
| assignment | 60s | optional | 2 attempts, 5s initial backoff |
| feedback projection | 2m | optional | 3 attempts, 5s initial backoff |

Non-retryable error categories:

1. invalid input or malformed config
2. cluster too small
3. missing prerequisites such as no centroids for assignment
4. unsupported source kinds

### 11.9 Data reference strategy

Externalize large data by reference using this pattern:

| Payload type | Reference shape |
|---|---|
| parsed documents | `duckdb://{account_id}/{run_id}/agreements` |
| chunk manifests | `duckdb://{account_id}/{run_id}/chunks` |
| clause manifests | `duckdb://{account_id}/{run_id}/clauses` |
| chunk embeddings | `lance://{account_id}/chunks?run_id={run_id}` |
| summary embeddings | `lance://{account_id}/documents?run_id={run_id}` |
| centroids | `duckdb://{account_id}/{run_id}/cluster_centroids` |
| soft memberships | `artifact://{account_id}/{run_id}/soft_memberships.json` |
| cluster profiles | `artifact://{account_id}/{run_id}/profiles/{cluster_id}.json` |

The exact URI scheme is internal, but it must be stable and parseable by all workers.

### 11.10 Recommended workflow queries and signals

Expose these for API gateway integration:

| Workflow | Query or signal | Purpose |
|---|---|---|
| `PipelineRunWorkflow` | `get_status` | run summary for polling |
| `PipelineRunWorkflow` | `get_stage_progress` | stage-level UI progress |
| `PipelineRunWorkflow` | `cancel_run` | cancel long-running run |
| `ClusterProcessingWorkflow` | `get_cluster_status` | cluster-level drill-down |
| `DocumentAssignmentWorkflow` | `get_assignment_status` | async assignment polling |
| `DocumentReprocessWorkflow` | `get_reprocess_status` | document reprocess polling |

---

## 12. Agent handoff context

This section is for the next implementation agent. It is deliberately operational and should be treated as the starting brief.

### 12.1 Source files to read first

Read these in order before writing code:

1. `core/minimal_pipeline.py`
2. `docs/TEMPORAL_DECOMPOSITION_PLAN.md`
3. `docs/FRONTEND_API_TEMPORAL_MICROSERVICES_DESIGN.md`
4. `core/store.py`
5. `core/lancedb_store.py`
6. `core/feedback_store.py`
7. `core/personalization.py`
8. `core/cluster_assigner.py`
9. `core/rag_extractor.py`
10. `core/rlm_v3.py`

### 12.2 Architecture decisions already made

Do not reopen these unless implementation proves them wrong:

1. The service cut is `api-gateway`, `orchestrator`, `ingestion`, `embedding`, `clustering`, `discovery`, `extraction`, `assignment`, `feedback-personalization`, and `query-service`.
2. `_process_cluster()` is the child-workflow boundary.
3. Discovery and extraction are separate services and task queues.
4. The assignment hot path stays out of synchronous Temporal orchestration.
5. DuckDB and LanceDB remain the source of truth for the first migration.
6. Workflow payloads carry references, not large arrays or full text.

### 12.3 First files to create

The next agent should start with these files:

```text
hackathon/packages/contracts/workflows/pipeline.py
hackathon/packages/contracts/workflows/clusters.py
hackathon/packages/contracts/workflows/assignment.py
hackathon/packages/contracts/workflows/common.py
hackathon/services/orchestrator/workflows/pipeline_run.py
hackathon/services/orchestrator/workflows/cluster_processing.py
hackathon/workers/ingestion/activities.py
hackathon/workers/embedding/activities.py
hackathon/workers/clustering/activities.py
hackathon/workers/discovery/activities.py
hackathon/workers/extraction/activities.py
hackathon/workers/assignment/activities.py
```

### 12.4 Recommended first vertical slice

Build this slice first:

1. `PipelineRunWorkflow`
2. `ParseBatchDocumentsActivity`
3. `GenerateChunksActivity`
4. `EmbedCorpusActivity`
5. `ClusterCorpusActivity`
6. `PersistRunManifestActivity`

That slice proves:

1. workflow wiring
2. storage references
3. task queue separation
4. run status and manifest generation

Only after that should the next agent implement the Stage 4 cluster fan-out.

### 12.5 Stage 4 implementation order

When moving to Stage 4, implement in this order:

1. `LoadClusterProcessingContextActivity`
2. `DiscoverClusterSchemaActivity`
3. `PersistDiscoveredFieldsActivity`
4. `PersistRlmIntentsActivity`
5. `RunConfidenceGateActivity`
6. `ExtractClusterValuesActivity`
7. `BuildClusterProfileActivity`
8. `ClusterProcessingWorkflow`
9. `ClusterFanoutWorkflow`

### 12.6 Risks the next agent should watch

1. accidental large Temporal payloads from chunk rows or extraction results
2. worker code trying to share one process-global DuckDB connection
3. RLM V3 thread-local LM context getting lost inside activity implementations
4. synchronous assignment being routed through long-running Temporal execution
5. cluster-global state leaking between retries

### 12.7 Definition of done for the first implementation PR

The first implementation PR is done when:

1. shared workflow contracts exist in code
2. `PipelineRunWorkflow` runs end-to-end through clustering using adapter activities
3. the API gateway can create a workflow and poll run status
4. manifest output is persisted in the same shape expected by the current system
5. no workflow payload contains corpus-scale arrays or full-document text blobs

