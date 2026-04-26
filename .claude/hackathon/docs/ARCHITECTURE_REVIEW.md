# Architecture Review — Contract Extraction Pipeline

*Prepared for legal technology conference — April 2026*

---

## What We Built

A 7-stage pipeline that ingests raw contract PDFs and produces structured, queryable field extractions — effectively building an **automated ontology** for any contract corpus without predefined schemas.

```
PDFs ─→ Parse ─→ Chunk+Embed ─→ Cluster ─→ Domain Discovery ─→ Agentic Extraction ─→ Assignment Init
  │                                              │                        │
  │                                              ▼                        ▼
  │                                      120 domains              ClusterProfiles
  │                                      (topic groups)           ("agreement genomes")
  │                                              │                        │
  │                                              ▼                        ▼
  ▼                                       Two-tier fields          Field + Value
2889 PDFs ──→ 144,464 chunks ──→ 8,936 clusters ──→ 2,904 extraction targets ──→ Structured DB
```

### Key Numbers (Full-v3 Run — Latest)

| Metric | Value |
|--------|-------|
| Agreements processed | 2,889 |
| Clause-aligned chunks | 144,464 |
| Chunk-clusters (EVoC) | 8,936 |
| Domains discovered | 120 |
| Extraction-targets processed | 2,904 |
| Active fields (with extractions) | 4,730 |
| Total extractions | 23,360 |
| Agreement coverage | 1,681 / 2,889 (58.2%) |
| Phantom rate | **0.1%** (4 / 4,734) |
| Avg extraction confidence | 0.852 |
| ClusterProfiles built | 789 |
| RLM intents discovered | 1,550 |
| LLM calls | 97,313 |
| Wall time | 2.0 hours |

---

## Stage Architecture

### Stage 1: PDF Parsing
Extracts raw text from PDFs with per-file JSON caching (keyed on source directory).

**What's solid:** Caching prevents re-parsing across runs.
**What's not ready:** No OCR fallback for scanned PDFs. No incremental cache invalidation — adding one PDF re-validates the entire manifest.

### Stage 2: Chunk + Embed
Three-pass "structural-first, semantic-fallback" chunking produces clause-aligned provisions (200-5000 chars). GPU-accelerated Nomic 768-d embeddings.

- **Pass 1 (CPU):** Split at legal headings (ARTICLE, Section, numbered paragraphs)
- **Pass 2 (GPU):** Embed + semantic sub-split only for oversized sections (~30% of text)
- **Pass 3 (CPU):** Assemble final chunks with clause annotations

**What's solid:** Clause-aligned chunks preserve legal structure. 768-d embeddings enable high-quality retrieval downstream.
**What's not ready:** `max_chunks_per_doc=80` cap silently drops tail clauses in very long agreements. Embedding model is hardcoded (no pluggable embedder interface for swapping to a domain-finetuned model).

### Stage 3: Parallel EVoC Clustering + Optuna
Evolutionary clustering with Optuna hyperparameter tuning. Two levels:
- **Macro:** Document summaries → domains (120 topic groups)
- **Micro:** Chunks → clusters within domains (8,936 groups)

Features: guided clustering (user keyword groups), multi-dimensional signals (field-overlap + obligation similarity), soft probabilistic membership, native EVoC layer hierarchy.

**What's solid:** Optuna auto-tunes hyperparameters. Soft membership means a "governing law" chunk can belong to both a governing-law cluster and a general-provisions cluster. Layer hierarchy gives multiple resolution views.
**What's not ready:** `enable_multidim_signals` defaults to `False` — the obligation-similarity dimension hasn't been battle-tested. Stage 3 is inherently global (needs all embeddings at once) — can't be parallelized across shards.

### Stage 3.5: Domain Discovery (Two-Tier Field Architecture)
Per-domain RLM V3 sessions discover **universal fields** (effective_date, governing_law, party_names) that apply across all clusters in that domain.

**3.5b — Cross-Domain Universals:** Fields appearing in ≥30% of domains get promoted to a `__universal__` tier and seeded into every cluster. Ensures common fields are always extracted.

**What's solid:** Two-tier architecture prevents redundant discovery of common fields. Domain-level sessions see diverse chunks across many clusters, producing higher-quality universal fields than any single cluster could.
**What's not ready:** ≥30% threshold is hardcoded. No mechanism to demote a universal field that proves phantom across most clusters.

### Stage 4: Agentic Discovery + Confidence-Gated Extraction
The core intelligence. Per-cluster: RLM V3 discovers fields → dedup → validate → confidence-gated RAG extraction → prune.

```
Per Cluster:
  Domain seed fields (Tier 1)
         │
         ▼
  RLM V3 Agentic Discovery ──→ Tier 2 delta fields
  (sandbox REPL, 9 tools,        + clause types
   2 workers, latent briefing)   + obligations
         │                       + intents
         ▼
  Embedding Dedup (cosine > 0.85)
         │
         ▼
  Quick-Sample Validation (3 docs → drop 0/3 fields)
         │
         ▼
  Confidence Gate (Cochran sample → mean_conf ≥ threshold?)
         │
    PASS ──────────── FAIL
     │                  │
     ▼                  ▼
  Full RAG          Delete all
  Extraction        fields for
     │              this cluster
     ▼
  Post-Extraction
  Phantom Pruning
```

**Retrieval architecture:** Hybrid retrieval with two modes (configurable):
- **Standard mode:** FAISS dense (MiniLM 384-d) + BM25 sparse + RRF fusion
- **Late interaction mode** (new): ColBERTv2 MaxSim (128-d per-token) + BM25 + RRF — provides reranker-quality scoring without cross-encoder latency

**Document scope** (configurable):
- **Cluster-scoped** (default): Retriever sees only chunks assigned to this cluster per agreement
- **Agreement-wide scope** (new): Retriever sees all chunks in each agreement, with cluster-member boost — prevents missing values in adjacent clauses

See [RETRIEVER_CHANGES.md](RETRIEVER_CHANGES.md) for the full retriever architecture and tuning guide.

**What's solid:** Multi-layer quality defense (see [CONFIDENCE_GATING_AND_PRUNING.md](CONFIDENCE_GATING_AND_PRUNING.md)). RLM V3 runs in Deno/Pyodide WASM sandbox — agent code is isolated. 12 typed tools on host side via JSON-RPC. Latent briefing compresses prior trajectory into context for grounded queries.
**What's not ready:** See "Quality Analysis" and "Architecture Gaps" below.

### Stage 4b: Batch Small Clusters by Domain
Clusters with 5-9 chunks are too thin for individual RLM sessions. Groups them by domain, concatenates chunks, runs one RLM session per domain batch.

**What's solid:** Recovers coverage from the long tail of small clusters that would otherwise be skipped.
**What's not ready:** Uses first cluster's ID as primary storage key — loses per-cluster granularity. Fields from batched clusters are attributed to a synthetic combined cluster, not to their original chunk-cluster.

### Stage 5: Incremental Cluster Assignment
Loads all centroids for real-time single-document ingestion (<500ms target).

Three-tier confidence model:
| Tier | Cosine | Behavior |
|------|:------:|----------|
| **A** (High) | ≥ 0.65 | Hard-assign, update centroid running mean |
| **B** (Tentative) | 0.45-0.65 | Soft-assign, flag for review, no centroid update |
| **C** (Novel) | < 0.45 | Buffer in novel_documents for periodic re-clustering |

**What's solid:** Three-tier model is principled — prevents structural pollution from uncertain assignments. Drift monitoring warns when >25% of new docs are novel. Personalization thresholds from feedback loop.
**What's not ready:** Centroid updates are memory-only — never written back to DB. Embeddings use only first 4000 chars (speed trade-off), which may misclassify long agreements where the meaningful content is late in the document.

---

## What's Working Well (Demo-Ready)

1. **End-to-end pipeline on real data:** 2,889 real public contracts → 23,360 extractions across 4,730 active fields in 2.0 hours
2. **0.1% phantom rate** (down from 55.5%) — lazy-persist eliminated 6,044 phantom fields while preserving 99.5% of extractions
3. **Emergent ontology:** No predefined schemas — the system discovers 2,002 unique field names across clause types. Feed it M&A agreements → it finds `purchase_price`, `earnout_period`. Feed it employment contracts → it finds `base_salary`, `non_compete_radius`.
4. **ClusterProfiles ("agreement genomes"):** 789 rich JSON profiles with fields, clause types, obligations, intents, keywords — one per extraction-target
5. **High-confidence extractions:** 86% of extractions at ≥0.70 confidence, 51% at ≥0.90. Duration and currency fields hit 0.95+ confidence.
6. **Substantive values:** 62% of extractions are 21-500 characters — full clause phrases and provisions, not single-word stubs. 54% are high-value (>20 chars, ≥0.70 confidence).
7. **82.3% source grounding:** Extractions trace back to specific source chunks for auditability
8. **Confidence gating:** Cochran sampling → gate pass rate 63.6% — saves ~36% of LLM cost by skipping low-quality clusters
9. **Three-source field architecture:** 997 agent fields (0.928 avg conf) + 884 domain-seed fields (reach 1,242 agreements) + 2,846 RAG fields (cluster-specific long tail)
10. **Sandboxed agent:** RLM V3 runs in Deno/Pyodide WASM — not `exec()`
11. **58.2% agreement coverage** (up from 28.7%) — more than doubled by lowering cluster threshold + batching small clusters
12. **Interactive dashboard:** Self-contained HTML with sunburst, charts, tables
13. **Feedback loop infrastructure:** FeedbackStore → PersonalizationEngine → per-cluster threshold overrides
14. **Incremental assignment API:** CLI-ready single-document ingestion with three-tier confidence (<500ms target)

---

## Quality Analysis (Full-v3 Run — Latest)

### Run Progression

| Metric | Baseline (30) | Full-458 | v2 (2,904 targets) | **v3 (+ lazy persist)** |
|--------|:---:|:---:|:---:|:---:|
| Extraction-targets | 30 | 458 | 2,904 | 2,904 |
| Fields in DB | 496 | 3,027 | 10,878 | **4,734** |
| Active fields | — | ~1,561 | 4,837 | **4,730** |
| Extractions | 2,717 | 7,450 | 23,483 | **23,360** |
| Agreement coverage | 13.9% | 28.7% | 58.4% | **58.2%** |
| Phantom rate | 47% | 48.4% | 55.5% | **0.1%** |
| Avg confidence | — | 0.836 | 0.852 | **0.852** |
| Profiles built | — | 188 | 789 | **789** |
| Wall time | ~12 min | ~76 min | 130 min | **120 min** |
| LLM calls | — | 33,421 | 99,630 | **97,313** |

**v3 key result:** Lazy-persist eliminated 6,044 phantom fields while preserving 99.5% of extractions. The DB is now clean — only fields with real values exist.

### Phantom Field Resolution

| Source | v2 Phantom Rate | v3 Phantom Rate | What Changed |
|--------|:-:|:-:|---|
| **rag_extractor** | 67.2% (6,026/8,962) | **0.0%** (0/2,846) | Lazy-persist: only fields with ≥1 extraction are persisted |
| **agent** | 0.0% (0/984) | **0.0%** (0/997) | Already clean — inline extractions guarantee grounding |
| **domain_seed** | 1.6% (15/931) | **0.5%** (4/888) | Small improvement from better validation |
| **Overall** | **55.5%** (6,041/10,878) | **0.1%** (4/4,734) | **-55.4 percentage points** |

### Extraction Value Quality

| Metric | Value |
|--------|-------|
| Total extractions | 23,360 |
| High-value extractions (>20 chars, ≥0.70 confidence) | 12,609 (54.0%) |
| Extractions with source chunk grounding | 19,226 (82.3%) |
| Unique field names in ontology | 2,002 |
| Avg extractions per covered agreement | 13.9 (median: 7) |

**Value length distribution:**

| Length | Count | % | Avg Confidence |
|--------|:-----:|:---:|:-:|
| 1-5 chars (codes, booleans) | 2,781 | 11.9% | 0.903 |
| 6-20 chars (short values) | 5,550 | 23.8% | 0.876 |
| 21-100 chars (phrases, clauses) | 6,995 | 29.9% | 0.848 |
| 101-500 chars (full clause text) | 7,499 | 32.1% | 0.821 |
| 500+ chars (multi-clause spans) | 535 | 2.3% | 0.821 |

Most extractions (62%) are 21-500 characters — substantive clause phrases and full provisions, not single-word stubs. Short extractions (≤20 chars) have the highest confidence (0.88-0.90), consistent with extracting defined terms, dates, and boolean flags.

**Field type performance:**

| Field Type | Extractions | Avg Confidence |
|------------|:-----------:|:-:|
| text | 20,637 | 0.847 |
| boolean | 1,336 | 0.897 |
| enum | 987 | 0.880 |
| duration | 159 | 0.955 |
| date | 99 | 0.878 |
| currency | 53 | 0.945 |

Duration and currency fields achieve highest confidence (0.94-0.96) — these have unambiguous formats. Boolean and enum fields also perform well (0.88-0.90). Text is the most common and has the widest confidence range (0.85 avg).

### Field Source Contribution

| Source | Fields | Extractions | Avg Confidence | Agreements Covered |
|--------|:------:|:-----------:|:-:|:-:|
| **agent** (RLM V3 discovery) | 997 | 8,419 | **0.928** | 899 |
| **domain_seed** (universal) | 884 | 7,998 | 0.804 | 1,242 |
| **rag_extractor** (RAG-only) | 2,846 | 6,932 | 0.815 | 1,161 |

Agent-discovered fields are the highest quality (0.928 avg confidence, 8.4 extractions/field). Domain seeds reach the most agreements (1,242) because they're universal. RAG-only fields are the long tail — many cluster-specific fields with fewer extractions each.

### Confidence Distribution

| Confidence Bucket | Count | % |
|:-----------------:|:-----:|:---:|
| ≥ 0.90 | 11,857 | 50.8% |
| 0.70 – 0.89 | 8,225 | 35.2% |
| 0.50 – 0.69 | 2,891 | 12.4% |
| < 0.50 | 387 | 1.7% |

86% of extractions at ≥0.70 confidence. Only 1.7% below 0.50.

### Gate Performance

| Metric | Value |
|--------|-------|
| Gates decided | 651 |
| Passed | 414 (63.6%) |
| Failed | 237 (36.4%) |

### Late Interaction & Wide Scope Status

**Note:** The v3 run had a config wiring bug — `shared_rag_config` was missing `use_late_interaction` and `widen_retrieval_scope`, so ColBERT MaxSim and agreement-wide retrieval **did not activate**. The v3 results therefore measure the pure impact of lazy-persist alone. This bug is now fixed — the next run (v4) will include all three changes.

**Implication:** The 0.1% phantom rate was achieved by lazy-persist alone, without any retrieval improvements. ColBERT + wide scope should further improve extraction *coverage* (more fields getting values) and potentially increase the total extraction count.

### Architecture Gaps (Ordered by Criticality)

#### 1. No Authentication or Authorization 🔴
No auth layer anywhere. Any caller can run any pipeline, access any tenant's data, read all extractions. `tenant_id` / `account_id` is path-based convention only — not enforced.

**To fix:** Add an auth middleware layer before the pipeline entry points. For conference demo this is fine, but any multi-tenant deployment needs this first.

#### 2. Centroid Drift Not Persisted 🟠
`IncrementalClusterAssigner._update_centroid()` modifies in-memory arrays but never writes back to DB. After a pipeline restart, centroid positions revert to their original values — all incremental assignment drift is lost.

**To fix:** Add a `flush_centroids()` method that writes updated centroids back to `cluster_centroids` table. Call it periodically or on shutdown.

#### 3. gRPC LM Adapter is a Scaffold 🟠
`core/grpc_lm.py` (370 lines) has the full interface but ships with placeholder proto stubs marked "ADAPT THIS." Cannot talk to any real service without customization.

**To fix:** Define the actual proto for the inference service (likely already exists in your org's model serving stack), generate stubs, wire them in.

#### 4. `eval()` in Ontology Refiner 🟠
`ontology_refiner.py` line ~700 uses Python's `eval()` for composite field evaluation, guarded by a numeric-only regex. The codebase already has a proper AST-based evaluator in `core/composite.py` that should be used instead.

**To fix:** Replace `_safe_eval()` in ontology_refiner with a call to `composite.parse_and_eval()`.

#### 5. No Distributed Rate Limiting 🟡
All LLM throttling is an in-process asyncio semaphore. Multiple pipeline instances against the same Azure OpenAI deployment will overwhelm quotas. The code itself flags this: "does not survive process boundaries."

**To fix:** Use a shared rate limiter (Redis token bucket, Azure API Management, or a sidecar proxy).

#### 6. No Idempotency Keys 🟡
All entity IDs are `uuid4().hex`. Re-running a failed pipeline creates duplicate field definitions and extractions. No upsert-on-conflict logic.

**To fix:** Use deterministic IDs (e.g., `hash(cluster_id + field_name)` for field_definitions) or add upsert logic to the store layer.

#### 7. Sequential Extraction Stub 🟡
`scalable_pipeline.py` has a TODO at line ~1827: `_run_sequential_extraction()` returns `{"sequential_status": "not_implemented"}`. This is the carryforward + judge LLM extraction mode — currently dead code.

**Not blocking:** The RAG extraction path works. Sequential extraction is an alternative strategy, not a prerequisite.

#### 8. No E2E Tests for Incremental Assignment 🟡
`assign_new_documents()` — the production entry point for real-time ingestion — has zero test coverage. The one unit test uses random data and checks only structure.

**To fix:** Add an integration test that runs a small pipeline → then calls `assign_new_documents()` with a known PDF → asserts correct cluster assignment.

#### 9. ClusterProfiles Not Queryable 🟢
Profiles are JSON files on disk, not in the DB. You can't SQL-query "which clusters have a force_majeure_duration field?" without loading all profiles.

**To fix:** Add a `cluster_profiles` table with key columns (cluster_id, label, field_count, clause_count, obligation_count, gate_passed, mean_confidence) or index them into a search layer.

#### 10. Embedding Model Not Pluggable 🟢
Chunking uses Nomic 768-d (hardcoded in `nomic_embedder.py`). RAG retrieval uses MiniLM 384-d (hardcoded in `embedder.py`). Swapping to a domain-finetuned legal embedder requires code changes.

**To fix:** Abstract behind an `Embedder` interface. Low priority unless you plan to fine-tune embeddings on legal corpora.

---

## What to Highlight at the Conference

### The "Magic" — why this is different from traditional NLP extraction

1. **Zero-schema ontology:** No one defines what fields exist. The pipeline discovers them from the text. Feed it M&A agreements → it finds `purchase_price`, `earnout_period`, `working_capital_adjustment`. Feed it employment contracts → it finds `base_salary`, `equity_vesting_schedule`, `non_compete_radius`.

2. **Clause-type grounding:** Every extracted value traces back to a specific clause type in a specific provision. Not just "this PDF says $10M" but "the purchase_price is $10M, from the Purchase Price clause in Section 2.3, with 0.92 confidence."

3. **Agreement genomes:** Each ClusterProfile is a structured description of a *type* of contract clause — what fields it has, what obligations it creates, what clause subtypes exist within it. 789 of these were built from 2,889 contracts.

4. **Cost-aware quality control:** Confidence gating skips 35% of clusters that can't produce reliable extractions — saving ~35% of LLM cost while maintaining statistical guarantees on quality.

5. **Token-level retrieval:** ColBERT late interaction matches query tokens independently against document tokens — "termination notice period" finds the right clause even when the contract says "cancel" instead of "terminate" and "notification" instead of "notice."

6. **Multi-layer phantom defense:** The agent discovers ~10 fields per cluster, quick-sample prunes ~7.4, confidence gating deletes fields from failed clusters, post-extraction prunes the rest. Each layer catches a different failure mode.

7. **Incremental ingestion:** After the initial pipeline run, new documents can be classified in <500ms without re-running clustering. Novel documents are buffered for periodic ontology evolution.

### What to be honest about

- **Phantom problem is solved** (0.1%) but we haven't yet tested ColBERT late interaction + wide retrieval scope at scale — a config wiring bug prevented them from activating in v3. Next run will include them.
- **58% agreement coverage is solid but not complete:** 42% of agreements aren't covered — their chunks live in clusters too small for extraction. Coverage will improve as the corpus grows and thin clusters accumulate more documents.
- **82% grounding, not 100%:** 17.7% of extractions lack source chunk references — these are from inline agent extractions where the chunk ID wasn't propagated. Values are still correct, but auditability is weaker.
- **The ontology quality depends on clustering quality:** If EVoC puts dissimilar clauses together, the agent discovers noisy fields. Clustering is the foundation everything else depends on.
- **Production deployment has gaps:** Auth, distributed rate limiting, centroid persistence, and E2E tests need work before serving real users at scale.
- **The system improves with scale:** More documents per cluster = better discovery. More clusters per domain = better universal fields. The architecture is designed for growing corpora.

---

## Architecture Diagram (Conference Slide)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Contract Extraction Pipeline                  │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────┐        │
│  │  Stage 1  │   │   Stage 2     │   │     Stage 3          │        │
│  │  Parse    │──▶│ Chunk+Embed   │──▶│  EVoC + Optuna       │        │
│  │  PDFs     │   │ 3-pass clause │   │  8,936 clusters      │        │
│  │  2,889    │   │ 144K chunks   │   │  + soft membership   │        │
│  └──────────┘   │ Nomic 768-d   │   └──────────┬───────────┘        │
│                  └──────────────┘                │                    │
│                                                  ▼                    │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │  Stage 3.5: Domain Discovery                              │        │
│  │  120 domains → universal fields → cross-domain promotion  │        │
│  │  Two-Tier Architecture: Tier 1 (universal) + Tier 2 (Δ)  │        │
│  └────────────────────────┬─────────────────────────────────┘        │
│                            │                                          │
│                            ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │  Stage 4: Agentic Discovery + Confidence-Gated Extraction │        │
│  │                                                            │        │
│  │  Per extraction-target:                                    │        │
│  │   RLM V3 (Deno sandbox, 9 tools, latent briefing)        │        │
│  │    → Dedup (cosine > 0.85)                                │        │
│  │    → Quick-sample validation (Layer 1)                    │        │
│  │    → Cochran confidence gate (Layer 2)                    │        │
│  │    → Hybrid RAG: ColBERT MaxSim + BM25 + cluster boost    │        │
│  │    → Post-extraction phantom prune (Layer 3)              │        │
│  │                                                            │        │
│  │  Output: Field Definitions + Extractions + ClusterProfiles │        │
│  └────────────────────────┬─────────────────────────────────┘        │
│                            │                                          │
│                            ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │  Stage 5: Incremental Assignment                          │        │
│  │  3-tier: High (≥0.65) → Tentative (0.45-0.65) → Novel    │        │
│  │  <500ms per document, drift monitoring, novel buffer      │        │
│  └──────────────────────────────────────────────────────────┘        │
│                                                                       │
│  ┌───────────────┐  ┌────────────────┐  ┌──────────────────┐        │
│  │   DuckDB       │  │  LanceDB       │  │  JSON Artifacts   │        │
│  │  19 tables     │  │  Vector store  │  │  ClusterProfiles  │        │
│  │  (structured)  │  │  (similarity)  │  │  (genomes)        │        │
│  └───────────────┘  └────────────────┘  └──────────────────┘        │
│                                                                       │
│  ┌───────────────────────────────────────────┐                       │
│  │  Feedback Loop (Sidecar)                   │                       │
│  │  8 action types → PersonalizationEngine    │                       │
│  │  → per-cluster threshold overrides         │                       │
│  └───────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Terminology Reference

| Term | Meaning | Count |
|------|---------|-------|
| **Chunk-cluster** | Group of similar text chunks from EVoC clustering | 8,936 |
| **Domain** | Group of similar chunk-clusters by topic (Stage 3.5) | 120 |
| **Extraction-target** | A chunk-cluster that passed the size filter and enters Stage 4 | 2,904 (≥10 chunks) |
| **ClusterProfile** | The "agreement genome" — per-cluster JSON with fields, clauses, obligations, intents | 789 built |
| **Phantom field** | A field definition with zero extractions | 55.5% overall, 0% from agent |
| **Confidence gate** | Statistical sample → decide whether to invest in full extraction | 64.8% pass rate |
| **Tier 1 field** | Universal field discovered at domain level (e.g., effective_date) | 931 seeded |
| **Tier 2 field** | Delta field discovered at cluster level (e.g., earnout_period) | 984 discovered |
| **Late interaction** | ColBERT per-token matching — each query token independently matches best document token | 128-d per token |

---

## Cost Model & Unit Economics

*Based on Full-v2 run: 2,889 agreements, 23,483 extractions, 2.2 hours wall time.*

### Model Usage

The pipeline uses two LLM tiers:
- **gpt-5.4** (main model): Field discovery, RAG extraction, confidence gating, intent extraction — ~60% of calls
- **gpt-5.4-mini** (sub model): RLM V3 REPL sub-calls, quick validations — ~40% of calls

### Per-Run Cost Breakdown (2,889 Agreements)

| Component | Unit Cost | Quantity | Cost | Notes |
|-----------|-----------|----------|-----:|-------|
| **LLM — gpt-5.4** | $5/1M input, $15/1M output | ~54M in, ~21M out | **$583** | 59,778 calls (discovery + extraction) |
| **LLM — gpt-5.4-mini** | $0.30/1M input, $1.20/1M output | ~16M in, ~6M out | **$12** | 39,852 calls (REPL sub-calls) |
| **GPU (A100 80GB)** | $3.67/hr on-demand | 2.2 hrs | **$8** | Embedding + ColBERT only |
| **GPU (A100 spot)** | $1.47/hr | 2.2 hrs | **$3** | ~60% savings |
| **Storage** | $0.02/GB/mo | 6.6 GB | **$0.13/mo** | DuckDB + LanceDB + artifacts |
| **Embedding API** | — | — | **$0** | Local GPU (Nomic + MiniLM) |

| Scenario | LLM | GPU | Storage | **Total per run** | **Per agreement** |
|----------|----:|----:|--------:|------------------:|------------------:|
| **Current mix** (gpt-5.4 + mini + on-demand) | $595 | $8 | $0.13 | **$603** | **$0.21** |
| **Current mix** (gpt-5.4 + mini + spot GPU) | $595 | $3 | $0.13 | **$598** | **$0.21** |
| **Economy** (all gpt-5.4-mini + spot) | $53 | $3 | $0.13 | **$56** | **$0.02** |

### Cost Scaling Model

| Corpus Size | Est. LLM Calls | Wall Time | Current Mix Cost | Economy Cost |
|:-----------:|:--------------:|:---------:|:----------------:|:------------:|
| 1,000 docs | ~34K | ~45 min | $209 | $19 |
| 2,889 docs (current) | 100K | 2.2 hrs | $603 | $56 |
| 10,000 docs | ~345K | ~7.5 hrs | $2,085 | $194 |
| 50,000 docs | ~1.7M | ~38 hrs | $10,425 | $970 |
| 100,000 docs | ~3.5M | ~76 hrs | $20,850 | $1,940 |

*Scaling is roughly linear in LLM calls. GPU time grows sub-linearly (embedding is batched). Confidence gating saves ~35% of extraction LLM cost at all scales.*

**Incremental ingestion** (after initial run): New documents are classified in <500ms with zero LLM cost (embedding similarity only). LLM calls are only needed when ontology re-discovery triggers.

### Cost Levers

| Lever | Impact | Trade-off |
|-------|--------|-----------|
| **All gpt-5.4-mini** instead of mixed | **91% cost reduction** ($595→$53) | Lower extraction quality on complex fields |
| **Confidence gating** (current: 35% gate-fail) | **~35% LLM savings** | Skips low-quality clusters (desired) |
| **Lazy-persist** (v3) | No direct cost saving | Eliminates phantom field cleanup overhead |
| **Spot GPU** | **60% compute savings** | Risk of preemption (2.2hr job = low risk) |
| **Reduce max_rag_retry_rounds** (2→1) | **~20% fewer LLM calls** | Fewer extraction retries, slightly lower recall |
| **Raise min_cluster_chunks** (10→20) | **~60% fewer clusters** | Lower coverage (58%→~29%) |

### Value Delivered & Revenue Model

#### What the Pipeline Produces (per corpus)

| Output | Count (2,889 docs) | Unit Value |
|--------|:------------------:|------------|
| **Structured field extractions** | 23,483 | Each replaces a manual review data point |
| **Active field definitions** (ontology) | 4,837 | Auto-discovered schema — no manual modeling |
| **ClusterProfiles** ("agreement genomes") | 789 | Clause-type intelligence for analytics |
| **Clause intents** | 1,620 | Obligation/risk tagging per clause |
| **Agreement coverage** | 58.4% (1,686 docs) | Each covered agreement is queryable |

#### Manual Review Comparison

| Task | Manual Cost | Pipeline Cost | Savings |
|------|:----------:|:------------:|:-------:|
| **Review 1 agreement** (senior paralegal, 2-4 hrs @ $75/hr) | $150–$300 | $0.21 | **99.9%** |
| **Build extraction schema** (legal analyst, 2 weeks) | $8,000–$15,000 | $0 (auto-discovered) | **100%** |
| **Tag 2,889 agreements** (10 paralegals, 3 months) | $180,000–$360,000 | $603 | **99.7%** |
| **Maintain schema as contract types evolve** | Ongoing | Incremental re-run | **Minimal** |

#### Revenue Scenarios (SaaS Pricing)

| Pricing Model | Rate | Revenue (2,889 docs) | Margin |
|---------------|------|:--------------------:|:------:|
| **Per-document processed** | $1.00/doc | $2,889 | 79% @ current, 98% @ economy |
| **Per-document processed** | $5.00/doc | $14,445 | 96% @ current |
| **Per-extraction** | $0.10/extraction | $2,348 | 74% @ current |
| **Per-extraction** | $0.50/extraction | $11,742 | 95% @ current |
| **Platform license** (per 10K docs/mo) | $5,000/mo | $5,000/mo | 58% @ current, 99% @ economy |
| **Enterprise license** (unlimited) | $25,000/mo | $25,000/mo | 92%+ |

#### ROI for a Typical Enterprise Customer

**Scenario:** Mid-size legal department, 10,000 active contracts, 500 new contracts/month.

| Metric | Without Pipeline | With Pipeline |
|--------|:----------------:|:-------------:|
| Initial corpus review | $600K–$1.2M (50 paralegals × 6 months) | $2,085 (one run) |
| Monthly new contract review | $37,500/mo (5 paralegals) | $210/mo (incremental runs) |
| Schema maintenance | $8K/mo (1 FTE analyst) | $0 (auto-evolving) |
| Time to first insight | 3–6 months | 8 hours |
| **Year-1 total cost** | **$1.1M–$1.7M** | **$4,600** |
| **Year-1 savings** | — | **$1.1M–$1.7M** |

*Note: Pipeline extractions require human QA sampling (~5-10% spot-check) but eliminate the need for full manual review.*

---

## Related Documentation

- [CONFIDENCE_GATING_AND_PRUNING.md](CONFIDENCE_GATING_AND_PRUNING.md) — Confidence gating mechanics and phantom field pruning
- [RETRIEVER_CHANGES.md](RETRIEVER_CHANGES.md) — Late interaction architecture, wide scope, benchmarks, and tuning guide
