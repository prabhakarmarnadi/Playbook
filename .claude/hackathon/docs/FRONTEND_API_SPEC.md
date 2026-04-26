# Frontend API Specification

**Date:** April 23, 2026  
**Status:** Proposed API contract  
**Primary source of truth:** `core/minimal_pipeline.py`  
**Companion docs:** `docs/FRONTEND_API_TEMPORAL_MICROSERVICES_DESIGN.md`, `docs/TEMPORAL_DECOMPOSITION_PLAN.md`

---

## 1. Why this doc exists

The backend implementation is much richer than the current API surface. The existing API only exposes a small job launcher plus job status, while the product already supports document ingestion, ontology exploration, extraction review, search, risk analysis, feedback, personalization, virtual clusters, and incremental assignment.

This document is the dedicated frontend-facing API contract. It extracts the product-facing surfaces from the broader Temporal and microservices design so frontend and API work can proceed independently from orchestration implementation.

Assumption: the backend is a queued batch system. Frontend requests create jobs, poll or subscribe to progress, and operate against read models built from persisted results.

---

## 2. Product use cases the API must support

The frontend needs to support all of these workflows without talking directly to Temporal, DuckDB, or LanceDB.

### 2.1 Document intake

1. Drop a batch of PDFs into the system.
2. Upload documents directly from browser.
3. Submit references to documents already in blob storage or filesystem.
4. Validate a batch before processing.
5. Start a pipeline run with a chosen configuration profile.
6. Resume a previous run from a later stage.
7. Cancel, retry, or clone a run.
8. Track per-document failures, parse failures, and low-confidence routing.

### 2.2 Pipeline monitoring

1. Watch queued, running, completed, failed, canceled, and partial-success states.
2. See stage-level progress for parse, chunk, embed, cluster, discovery, extraction, intents, profile generation, and assigner initialization.
3. Inspect cluster-level progress during Stage 4 fan-out.
4. See run artifacts, manifest, timing, cost, LLM stats, and error summaries.

### 2.3 Taxonomy and ontology exploration

1. Browse domains, clusters, clauses, fields, composites, profiles, soft memberships, and low-confidence chunks.
2. Inspect cluster centroids, keywords, obligations, clause types, and extraction coverage.
3. View the ontology tree: domain -> cluster -> field -> composite.
4. Query cluster layers and alternate clustering views when debugging clustering quality.

### 2.4 Extraction review and authoring

1. Review example sets before scaling to full extraction.
2. Approve, reject, or correct extracted values.
3. Create, update, and delete fields.
4. Add gold examples and field descriptions.
5. Create and delete composite fields.
6. Re-run extraction for a cluster, agreement, or single chunk.

### 2.5 Search, analytics, and graph flows

1. Hybrid search over chunks.
2. Knowledge graph exploration and recommendations.
3. Risk scoring and portfolio views.
4. Dashboard metrics, health metrics, and evaluation regressions.

### 2.6 Feedback and personalization

1. Correct cluster assignments.
2. Create virtual clusters.
3. Promote virtual clusters.
4. Merge or split clusters.
5. Rename clusters, fields, and intents.
6. Approve or reject fields, extractions, clusters, clause names, domains, or relationships.
7. Tune graph edges.
8. Override intent labels per tenant.
9. Adjust thresholds and auto-tune them from feedback.
10. Export and audit feedback history.

### 2.7 Incremental and real-time ingestion

1. Submit one new document for assignment to existing clusters.
2. Submit many new documents for assignment.
3. Route documents to high-confidence, tentative-review, or novel buckets.
4. Surface novel documents for later re-clustering.

---

## 3. API design principles

### 3.1 External API shape

1. REST for command and query APIs.
2. Server-sent events or WebSocket for job progress.
3. Idempotent create endpoints via `Idempotency-Key`.
4. Tenant-aware requests via auth claims plus explicit `tenant_id` or `account_id`.
5. Long-running operations always return a job or workflow handle.

### 3.2 Backend separation

The frontend should speak only to an API gateway or backend-for-frontend layer. That layer will:

1. Validate input.
2. Translate commands into Temporal workflow starts or signals.
3. Query read models for dashboards and drill-down views.
4. Normalize the current DuckDB and LanceDB-backed data into stable response contracts.

### 3.3 Read/write split

Use two families of APIs:

1. Command APIs: mutate state or start work. These are async and queue-backed.
2. Query APIs: read persisted state, dashboards, taxonomy, and review queues.

---

## 4. Canonical entities the frontend needs

The API contracts should make these entities first-class:

| Entity | Description | Backing source today |
|---|---|---|
| `document_batch` | A user-submitted group of documents | New command layer entity |
| `document` | A PDF or parsed agreement | `agreements` + ingestion metadata |
| `pipeline_run` | End-to-end processing run | `pipeline_runs` + manifest |
| `run_stage` | Stage execution summary | Manifest + workflow history |
| `domain` | Macro document grouping | `domains` |
| `cluster` | Micro clause-type cluster | `clusters` |
| `cluster_profile` | Cluster genome / extraction profile | JSON artifacts |
| `cluster_layer` | Alternate clustering layer metadata | `cluster_layers`, `cluster_layer_meta` |
| `chunk` | Clause-aligned text unit | `chunks` |
| `clause` | Structural clause record | `clauses` |
| `field` | Field definition for a cluster | `field_definitions` |
| `composite` | Derived field definition | `composite_definitions` |
| `extraction` | Extracted field value | `extractions` |
| `intent_type` | Canonical intent taxonomy node | `intent_types` |
| `clause_intent` | Intent found on a clause | `clause_intents` |
| `assignment` | Incremental assignment decision | runtime result + stored novel buffer |
| `feedback_event` | User correction or approval | `feedback_events` |
| `virtual_cluster` | Tenant-specific cluster overlay | `virtual_clusters` |
| `personalization_config` | Thresholds and tuning | `tenant_config` |
| `review_item` | Low-confidence or human-review task | New read model derived from multiple tables |

---

## 5. Frontend API surface

This section describes the full external API the frontend needs. The current server exposes only a small subset; the following is the target contract.

All routes below are shown under `/v1`.

### 5.1 Batches and document drop APIs

These APIs exist so the UI can support drag-and-drop, staged upload, validation, and queued processing.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/document-batches` | Create an empty batch and reserve IDs |
| `POST` | `/document-batches/{batch_id}/files` | Upload one or more PDF files |
| `POST` | `/document-batches/{batch_id}/sources` | Register blob URLs, S3 URIs, or server-side file paths |
| `POST` | `/document-batches/{batch_id}/validate` | Validate file types, duplicates, size, readability |
| `GET` | `/document-batches/{batch_id}` | Get batch metadata and counts |
| `GET` | `/document-batches/{batch_id}/documents` | List documents in the batch |
| `DELETE` | `/document-batches/{batch_id}` | Delete an unused batch |
| `POST` | `/document-batches/{batch_id}/start-run` | Start a pipeline run from this batch |

#### Request shape: `POST /document-batches/{batch_id}/start-run`

```json
{
  "account_id": "acct_public_contracts",
  "tenant_id": "tenant_acme_legal",
  "config_profile": "production_rlm_v3_unified",
  "config_overrides": {
    "device": "cuda",
    "max_docs": 500,
    "discovery_mode": "rlm_v3_unified",
    "enable_field_features": true,
    "enable_confidence_gate": true,
    "confidence_gate_threshold": 0.65,
    "enable_soft_membership": true,
    "enable_multidim_signals": true,
    "guided_keywords": {
      "termination": ["termination", "cancellation", "expiry"],
      "payment": ["payment", "fees", "invoice"]
    }
  },
  "resume_from_stage": null,
  "priority": "normal",
  "notify": {
    "on_complete": true,
    "on_failure": true
  }
}
```

### 5.2 Pipeline run and job APIs

These are the APIs every dashboard and progress screen will use.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/runs` | Create a run directly without pre-created batch |
| `GET` | `/runs` | List runs with filters |
| `GET` | `/runs/{run_id}` | Get run summary |
| `GET` | `/runs/{run_id}/stages` | Get stage-by-stage progress and timing |
| `GET` | `/runs/{run_id}/events` | Get ordered status events |
| `GET` | `/runs/{run_id}/artifacts` | List manifests, cluster profiles, exports |
| `GET` | `/runs/{run_id}/errors` | List errors and failed items |
| `POST` | `/runs/{run_id}/cancel` | Cancel run |
| `POST` | `/runs/{run_id}/retry` | Retry failed activities or entire run |
| `POST` | `/runs/{run_id}/resume` | Resume from a specified stage |
| `POST` | `/runs/{run_id}/clone` | Clone config into a new run |
| `GET` | `/runs/{run_id}/stream` | SSE stream of progress updates |

### 5.3 Run configuration and profiles APIs

The pipeline has many keyword arguments and feature flags. The frontend needs profile-driven configuration, not a raw unstructured form.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/config/profiles` | List named run profiles |
| `GET` | `/config/profiles/{profile_id}` | Get defaults and allowed overrides |
| `POST` | `/config/profiles` | Create tenant-specific profile |
| `PATCH` | `/config/profiles/{profile_id}` | Update profile |
| `GET` | `/config/schema` | Return typed config schema grouped by category |

The config schema should group fields into:

1. Ingestion
2. Chunking
3. Embedding
4. Clustering
5. Discovery
6. Extraction
7. Feedback and personalization
8. Confidence gating
9. Routing and assignment
10. Storage and artifact locations

### 5.4 Documents, agreements, chunks, and clauses APIs

The frontend needs document-level drill-down independent of runs.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/documents` | List processed documents |
| `GET` | `/documents/{document_id}` | Document summary |
| `GET` | `/documents/{document_id}/text` | Parsed raw text |
| `GET` | `/documents/{document_id}/chunks` | Chunk list for document |
| `GET` | `/documents/{document_id}/clauses` | Clause list for document |
| `GET` | `/documents/{document_id}/assignments` | Cluster assignments for document chunks |
| `GET` | `/documents/{document_id}/extractions` | Extractions grouped by field |
| `GET` | `/documents/{document_id}/intents` | Clause intents for document |
| `POST` | `/documents/{document_id}/reprocess` | Requeue document through selected stages |
| `POST` | `/documents/{document_id}/reassign` | Re-run incremental assignment |

### 5.5 Taxonomy, clusters, and ontology APIs

These power domains, cluster explorer, ontology tree, cluster map, and low-confidence review.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/domains` | List domains |
| `GET` | `/domains/{domain_id}` | Get domain detail |
| `GET` | `/domains/{domain_id}/clusters` | List clusters in domain |
| `GET` | `/clusters` | List clusters with filters |
| `GET` | `/clusters/{cluster_id}` | Get cluster detail |
| `GET` | `/clusters/{cluster_id}/profile` | Get cluster profile artifact |
| `GET` | `/clusters/{cluster_id}/chunks` | List chunk members |
| `GET` | `/clusters/{cluster_id}/clauses` | List clauses represented in cluster |
| `GET` | `/clusters/{cluster_id}/agreements` | List agreements contributing to cluster |
| `GET` | `/clusters/{cluster_id}/fields` | List fields |
| `GET` | `/clusters/{cluster_id}/extractions` | List extractions in cluster |
| `GET` | `/clusters/{cluster_id}/intents` | List discovered intents |
| `GET` | `/clusters/{cluster_id}/soft-memberships` | Soft assignment distribution |
| `GET` | `/clusters/{cluster_id}/outliers` | Outlier or low-confidence chunks |
| `GET` | `/clusters/{cluster_id}/keywords` | BM25 and guided keywords |
| `GET` | `/clusters/{cluster_id}/coverage` | Extraction and field coverage |
| `GET` | `/ontology/tree` | Domain -> cluster -> field -> composite tree |
| `GET` | `/ontology/graph` | Graph view payload for UI |
| `GET` | `/cluster-layers` | List clustering runs and selected layers |
| `GET` | `/cluster-layers/{clustering_run}` | Get layer metadata |
| `GET` | `/cluster-layers/{clustering_run}/{layer_index}` | Get item assignments for layer |

### 5.6 Cluster curation APIs

These are command APIs for ontology managers and reviewers.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/clusters/{cluster_id}/rename` | Rename cluster and description |
| `POST` | `/clusters/{cluster_id}/approve` | Mark cluster approved |
| `POST` | `/clusters/{cluster_id}/reject` | Mark cluster rejected or deprecated |
| `POST` | `/clusters/merge` | Merge multiple clusters |
| `POST` | `/clusters/split` | Split a cluster by selected members or rules |
| `POST` | `/clusters/{cluster_id}/recompute-profile` | Rebuild cluster profile |
| `POST` | `/clusters/{cluster_id}/rerun-discovery` | Re-run field discovery |
| `POST` | `/clusters/{cluster_id}/rerun-extraction` | Re-run extraction |
| `POST` | `/clusters/{cluster_id}/members/reassign` | Reassign chunks to a different cluster |
| `POST` | `/clusters/{cluster_id}/promote-to-domain` | Optional future action for macro taxonomy changes |

### 5.7 Fields, composites, and extraction review APIs

These power field CRUD, example-set review, extraction correction, and composite fields.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/clusters/{cluster_id}/example-sets` | Generate example set for review |
| `GET` | `/example-sets/{example_set_id}` | Get example set and candidate extractions |
| `POST` | `/example-sets/{example_set_id}/approve` | Scale extraction to full cluster |
| `POST` | `/example-sets/{example_set_id}/reject` | Reject example set and stop scaling |
| `GET` | `/fields/{field_id}` | Field detail |
| `POST` | `/fields` | Create field |
| `PATCH` | `/fields/{field_id}` | Update field |
| `DELETE` | `/fields/{field_id}` | Delete field |
| `POST` | `/fields/{field_id}/examples` | Add gold example |
| `DELETE` | `/fields/{field_id}/examples/{example_id}` | Remove example |
| `GET` | `/fields/{field_id}/coverage` | Coverage summary |
| `POST` | `/composites` | Create composite field |
| `DELETE` | `/composites/{field_id}` | Delete composite |
| `GET` | `/composites/{field_id}/preview` | Preview computed values |
| `POST` | `/extractions/{extraction_id}/correct` | Correct a value |
| `POST` | `/extractions/{extraction_id}/approve` | Approve extraction |
| `POST` | `/extractions/{extraction_id}/reject` | Reject extraction |
| `POST` | `/extractions/bulk-correct` | Bulk grid edits from table UI |
| `GET` | `/extractions` | List extractions with filters |

### 5.8 Intents and obligation APIs

RLM V3 and intent extraction produce separate intent-level objects that need their own review surface.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/intent-types` | List canonical intent types |
| `GET` | `/intent-types/{intent_type_id}` | Intent type detail |
| `GET` | `/clause-intents` | Query clause intents |
| `GET` | `/clause-intents/{intent_id}` | Clause intent detail |
| `POST` | `/clause-intents/{intent_id}/correct` | Correct label, summary, or parties |
| `POST` | `/clause-intents/{intent_id}/approve` | Approve intent |
| `POST` | `/clause-intents/{intent_id}/reject` | Reject intent |
| `POST` | `/intent-overrides` | Create tenant-specific override |
| `PATCH` | `/intent-overrides/{override_id}` | Update override |
| `DELETE` | `/intent-overrides/{override_id}` | Remove override |
| `GET` | `/intent-overrides` | List overrides |

### 5.9 Search, graph, risk, and analytics APIs

These are query-heavy APIs used by dashboard, search, risk, and knowledge graph views.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/search/chunks` | Hybrid chunk search |
| `GET` | `/search/documents` | Semantic document search |
| `GET` | `/search/suggest` | Query suggestions |
| `GET` | `/knowledge-graph/summary` | Top-level KG stats |
| `GET` | `/knowledge-graph/archetypes` | Deal archetypes |
| `GET` | `/knowledge-graph/recommendations` | Missing intent recommendations |
| `GET` | `/knowledge-graph/anomalies` | Anomaly detection results |
| `GET` | `/knowledge-graph/rules` | Implication rules |
| `GET` | `/risk/summary` | Portfolio risk summary |
| `GET` | `/risk/agreements` | Agreement risk list |
| `GET` | `/risk/agreements/{agreement_id}` | Risk profile per agreement |
| `POST` | `/risk/score` | Queue heuristic or LLM risk scoring |
| `GET` | `/analytics/dashboard` | Dashboard metrics |
| `GET` | `/analytics/health` | Pipeline and data health |
| `POST` | `/analytics/evaluations` | Run eval job |
| `GET` | `/analytics/evaluations/{evaluation_id}` | Get eval results |

### 5.10 Feedback and personalization APIs

This section must cover every correction flow the frontend exposes.

#### Generic audit-friendly event API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/feedback/events` | Record a typed feedback event |
| `GET` | `/feedback/events` | Query history |
| `GET` | `/feedback/summary` | Aggregate counts, weights, recent actions |
| `GET` | `/feedback/export` | Export CSV or JSON |

#### Specialized feedback APIs

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/feedback/cluster-corrections` | Reassign chunk or clause to cluster |
| `POST` | `/feedback/cluster-merges` | Merge clusters |
| `POST` | `/feedback/cluster-splits` | Split cluster |
| `POST` | `/feedback/cluster-renames` | Rename cluster |
| `POST` | `/feedback/domain-renames` | Rename domain |
| `POST` | `/feedback/field-corrections` | Rename, delete, or change field type |
| `POST` | `/feedback/extraction-corrections` | Correct extracted value |
| `POST` | `/feedback/intent-corrections` | Correct intent label/summary |
| `POST` | `/feedback/edge-corrections` | Link or unlink KG edges |
| `POST` | `/feedback/threshold-adjustments` | Save thresholds |
| `POST` | `/feedback/virtual-clusters` | Create tenant virtual cluster |
| `POST` | `/feedback/virtual-clusters/{cluster_id}/promote` | Promote virtual cluster |
| `GET` | `/feedback/virtual-clusters` | List virtual clusters |
| `GET` | `/feedback/intent-overrides` | List current intent overrides |
| `POST` | `/feedback/auto-tune` | Recompute personalization thresholds |
| `GET` | `/feedback/config` | Read tenant threshold config |
| `PATCH` | `/feedback/config` | Update tenant threshold config |

### 5.11 Review queues APIs

The frontend needs explicit queues for human review. These are not tables today, but should be materialized from low-confidence memberships, tentative assignments, failed extractions, and user-generated worklists.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/review-queues` | List queue definitions and counts |
| `GET` | `/review-queues/{queue_name}/items` | List items |
| `POST` | `/review-queues/{queue_name}/items/{item_id}/claim` | Claim review item |
| `POST` | `/review-queues/{queue_name}/items/{item_id}/resolve` | Resolve item |
| `POST` | `/review-queues/{queue_name}/items/{item_id}/requeue` | Return item to queue |

Recommended built-in queues:

1. `tentative_assignments`
2. `novel_documents`
3. `low_confidence_chunks`
4. `failed_documents`
5. `failed_clusters`
6. `extraction_review`
7. `intent_review`
8. `taxonomy_review`

### 5.12 Incremental assignment APIs

These support the real-time path without forcing the frontend to understand centroids or embedder state.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/assignments` | Assign one document |
| `POST` | `/assignments/batch` | Assign many documents |
| `GET` | `/assignments/{assignment_id}` | Assignment detail |
| `GET` | `/assignments` | List assignments |
| `POST` | `/assignments/{assignment_id}/confirm` | Confirm tentative assignment |
| `POST` | `/assignments/{assignment_id}/override` | Force assignment to another cluster |
| `POST` | `/assignments/{assignment_id}/mark-novel` | Keep in novel buffer |

#### Response shape: `POST /assignments`

```json
{
  "assignment_id": "asg_01HV...",
  "document_id": "doc_01HV...",
  "cluster_id": "cluster_termination_12",
  "cluster_label": "termination rights",
  "confidence": 0.88,
  "assignment_type": "high_confidence",
  "top_matches": [
    {"cluster_id": "cluster_termination_12", "score": 0.88},
    {"cluster_id": "cluster_expiry_04", "score": 0.72},
    {"cluster_id": "cluster_notice_09", "score": 0.55}
  ],
  "review_required": false
}
```

---

## 6. Feedback taxonomy

The current feedback sidecar already supports several actions and entity types, but the frontend needs a broader and more stable taxonomy that can map to current and future flows.

### 6.1 Canonical feedback entity types

| Entity type | Existing support | Needed by frontend |
|---|---|---|
| `document` | partial | document reprocess, assignment override, failure triage |
| `chunk` | indirect | chunk-level reassignments and review |
| `clause` | partial | clause-level intent and taxonomy review |
| `domain` | yes | rename, approve, reject |
| `cluster` | yes | merge, split, rename, approve, reject |
| `virtual_cluster` | yes | create, promote, assign |
| `field` | yes | create, update, approve, reject, delete |
| `composite_field` | no | derived-field approval and deletion |
| `extraction` | indirect | correct, approve, reject |
| `intent` | yes | relabel, approve, reject, override |
| `edge` | yes | link, unlink, reweight |
| `threshold` | yes | tuning and auto-tune tracking |
| `run` | no | rerun, cancel, retry, annotate |

### 6.2 Canonical feedback action types

| Action | Meaning |
|---|---|
| `create` | user created a new entity |
| `approve` | entity or output accepted |
| `reject` | entity or output rejected |
| `correct` | value changed from old to new |
| `rename` | display name or label changed |
| `retype` | field or intent type changed |
| `assign` | item assigned to taxonomy node |
| `unassign` | assignment removed |
| `merge` | multiple entities collapsed |
| `split` | one entity divided into multiple |
| `link` | KG edge or relation strengthened |
| `unlink` | KG edge or relation weakened/removed |
| `tune` | threshold or weight adjusted |
| `promote` | virtual or provisional item promoted |
| `demote` | approved item downgraded or reopened |
| `reprocess` | item queued for another backend pass |

### 6.3 Generic feedback event contract

The generic event API should allow specialized UI screens to map cleanly onto one write model.

```json
{
  "tenant_id": "tenant_acme_legal",
  "user_id": "user_123",
  "entity_type": "cluster",
  "entity_id": "cluster_termination_12",
  "action": "split",
  "old_value": {
    "label": "termination rights"
  },
  "new_value": {
    "children": [
      {"label": "termination for cause"},
      {"label": "termination for convenience"}
    ]
  },
  "context": {
    "run_id": "run_ab12cd34",
    "reason": "cluster mixes cause and convenience language",
    "selected_chunk_ids": ["chunk_1", "chunk_2", "chunk_3"]
  },
  "source_document_id": null,
  "session_id": "sess_456"
}
```

### 6.4 Frontend feedback screens mapped to canonical events

| UI action | Canonical entity | Canonical action |
|---|---|---|
| Correct cluster assignment | `chunk` or `clause` | `assign` |
| Create virtual cluster | `virtual_cluster` | `create` |
| Promote virtual cluster | `virtual_cluster` | `promote` |
| Merge clusters | `cluster` | `merge` |
| Split cluster | `cluster` | `split` |
| Rename domain or cluster | `domain` or `cluster` | `rename` |
| Correct field schema | `field` | `correct` or `retype` |
| Delete field | `field` | `reject` |
| Correct extracted value | `extraction` | `correct` |
| Approve extraction | `extraction` | `approve` |
| Override intent label | `intent` | `rename` |
| Link or unlink graph edge | `edge` | `link` or `unlink` |
| Save thresholds | `threshold` | `tune` |
| Requeue document | `document` | `reprocess` |

---

## 7. Eventing and progress model for the frontend

The frontend needs a consistent async model.

### 7.1 Command response pattern

Every async write endpoint should return:

```json
{
  "job_id": "job_01HV...",
  "workflow_id": "run/acct_public_contracts/run_ab12cd34",
  "status": "queued",
  "submitted_at": "2026-04-23T18:00:00Z",
  "links": {
    "job": "/v1/runs/run_ab12cd34",
    "stream": "/v1/runs/run_ab12cd34/stream"
  }
}
```

### 7.2 Progress event model

SSE messages should use a normalized shape:

```json
{
  "event_type": "stage_progress",
  "run_id": "run_ab12cd34",
  "stage": "discovery_extraction",
  "status": "running",
  "completed": 24,
  "total": 188,
  "message": "processing cluster 25 of 188",
  "timestamp": "2026-04-23T18:05:22Z"
}
```

Other event types:

1. `run_queued`
2. `run_started`
3. `stage_started`
4. `stage_progress`
5. `cluster_progress`
6. `review_item_created`
7. `run_completed`
8. `run_failed`
9. `run_canceled`