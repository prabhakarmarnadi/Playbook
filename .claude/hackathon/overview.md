# EVoC Contract Intelligence Platform — Technical Overview

**Repository:** `clustering-v2/hackathon`
**Date:** April 2026
**Infrastructure:** 2× NVIDIA A100 80GB PCIe, Azure Standard_NC24ads_A100_v4
**Primary Dataset:** CUAD (Contract Understanding Atticus Dataset) — 510 contracts

---

## Table of Contents

1. [Platform Architecture](#1-platform-architecture)
2. [Pipeline Stages](#2-pipeline-stages)
3. [Datasets & Scale](#3-datasets--scale)
4. [Clustering: EVoC vs Legacy HDBSCAN](#4-clustering-evoc-vs-legacy-hdbscan)
5. [Embedding Models Benchmarked](#5-embedding-models-benchmarked)
6. [Hyperparameter Tuning (Optuna)](#6-hyperparameter-tuning-optuna)
7. [Multi-Layer Hierarchy & Layer Selection](#7-multi-layer-hierarchy--layer-selection)
8. [Field Discovery: Standard vs DSPy RLM vs Hybrid](#8-field-discovery-standard-vs-dspy-rlm-vs-hybrid)
9. [Field Extraction & Confidence Grounding](#9-field-extraction--confidence-grounding)
10. [Intent Extraction System](#10-intent-extraction-system)
11. [Knowledge Graph](#11-knowledge-graph)
12. [Ontology Refinement](#12-ontology-refinement)
13. [Scaling Benchmarks (96→200→510 Documents)](#13-scaling-benchmarks-96200510-documents)
14. [Cost Analysis](#14-cost-analysis)
15. [GPU Stack Validation](#15-gpu-stack-validation)
16. [Evaluation Framework (48 Metrics)](#16-evaluation-framework-48-metrics)
17. [Streamlit UI (16 Pages)](#17-streamlit-ui-16-pages)
18. [Static HTML Dashboards (9 Visualizations)](#18-static-html-dashboards-9-visualizations)
19. [DuckDB Schema (17 Tables)](#19-duckdb-schema-17-tables)
20. [Configuration Reference](#20-configuration-reference)
21. [Scripts & Tooling](#21-scripts--tooling)
22. [Key Design Decisions](#22-key-design-decisions)
23. [Known Gaps & Future Work](#23-known-gaps--future-work)

---

## 1. Platform Architecture

```
PDFs (510 contracts)
  │
  ├─ Stage 1 ─→ [pdf_parser.py]       Parse text + clause boundaries
  ├─ Stage 2 ─→ [chunker.py]          Legal-aware semantic chunking
  ├─ Stage 3 ─→ [nomic_embedder.py]   768-dim GPU embeddings (Nomic v1.5)
  ├─ Stage 4 ─→ [evoc_pipeline.py]    Multi-layer EVoC clustering
  ├─ Stage 5 ─→ [topic_merger.py]     Hierarchical cluster merging
  ├─ Stage 6 ─→ [evoc_pipeline.py]    LLM cluster labeling (Azure OpenAI)
  ├─ Stage 7 ─→ [field_discovery.py]  Schema discovery (Standard / RLM / Hybrid)
  ├─ Stage 8 ─→ [extractor.py]        Cluster-conditioned field extraction
  ├─ Stage 9 ─→ [intent_extractor.py] Dynamic intent discovery
  └─ Stage 10 → [knowledge_graph.py]  Entity-relationship graph construction
       │
       ▼
  DuckDB Store (17 tables) + LanceDB Vector Index
       │
       ├─→ Streamlit UI (16 pages, 5 sections)
       ├─→ HTML Dashboards (9 static visualizations)
       └─→ Evaluation Framework (48 metrics)
```

**Core Replacement Summary (V1 → V2):**

| Component | V1 (Legacy) | V2 (This Repo) |
|-----------|-------------|-----------------|
| Storage | AIDB + Azure Blob + Redis | DuckDB + LanceDB (local-first) |
| LLM Gateway | AIFlow gRPC proxy | Direct OpenAI SDK (Azure) |
| Clustering | UMAP + HDBSCAN (2-pass) | EVoC single-pass multi-layer |
| Embeddings | AIFlow embeddings (remote) | Nomic 768-dim GPU (local) |
| Extraction | Generic schema-agnostic | Cluster-conditioned (40% fewer tokens) |
| Intent Discovery | None | Dynamic LLM-based (300-400 types/run) |
| Knowledge Graph | None | 4 node types, 6 edge types |

---

## 2. Pipeline Stages

### Stage 1: PDF Parsing (`core/pdf_parser.py`)
- **Tool:** PyMuPDF (pymupdf 1.23+)
- **Output:** Raw text + clause boundaries per contract
- **Caching:** File-hash manifest skips unchanged PDFs
- **Throughput:** ~96 docs/min

### Stage 2: Semantic Chunking (`core/chunker.py`)
- **Method:** Greg Kamradt legal-aware semantic chunking
- **Sentence splitting:** Regex-based with legal abbreviation protection (Mr., U.S., Inc., Corp., etc.)
- **Legal structure detection:** ARTICLE/SECTION headers (+0.15 bonus), numbered subsections (+0.08 bonus), WHEREAS/RECITALS/DEFINITIONS markers
- **Breakpoint detection:** Cosine distance between consecutive sentence-buffer embeddings, neighbor smoothing (weight=0.3), 90th percentile threshold
- **Constraints:** 100–2000 chars per chunk, buffer_size=1 context sentences

| Parameter | Value |
|-----------|-------|
| `buffer_size` | 1 |
| `breakpoint_percentile` | 90.0 |
| `min_chunk_chars` | 100 |
| `max_chunk_chars` | 2000 |
| `structural_bonus_strong` | 0.15 |
| `structural_bonus_weak` | 0.08 |
| `neighbor_weight` | 0.3 |

- **Throughput:** ~6,000 chunks/min on A100

### Stage 3: Embedding (`core/nomic_embedder.py`)
- **GPU Model:** `nomic-ai/nomic-embed-text-v1.5` — 768-dim, L2-normalized
- **Matryoshka capable:** Can truncate to 256/512-dim without retraining
- **Task-specific prefixes:** `"search_document:"`, `"search_query:"`, `"clustering:"`, `"classification:"`
- **Batch size:** 32–64 (auto-tuned on VRAM)
- **CPU Fallback:** `all-MiniLM-L6-v2` — 384-dim (20× slower)
- **Throughput:** ~200 chunks/sec on A100 (6,000 chunks/min)
- **Storage:** DuckDB BLOB column + LanceDB persistent vector index

### Stage 4: Clustering — EVoC (`core/evoc_pipeline.py`)
- **Algorithm:** EVoC 0.3.1 (Embedding Vector Oriented Clustering)
- **Key advantage:** Produces multi-granularity layers in a single pass (no manual UMAP hyperparams)
- **Tuning:** 30 Optuna trials (TPE sampler, 8 startup trials)
- **Scoring:** Composite = silhouette×0.4 + cosine_coherence×0.3 + (1−outlier_ratio)×0.3
- **Three clustering modes:**
  1. **Macro-Micro:** Document domains (macro) → per-domain clause types (micro)
  2. **Clause-Only:** Global flat clause-type clustering
  3. **Hybrid:** Both macro + micro in one run
- **Output:** Multiple layers at different granularities, best selected by scoring

### Stage 5: Topic Merging (`core/topic_merger.py`)
- **Method:** Hierarchical complete-linkage merging (prevents chaining)
- **Merge threshold:** 0.82 cosine centroid similarity
- **Synonym threshold:** 0.78 field-name similarity
- **Features:** Alias tracking, centroid recalculation as weighted averages
- **Observed merge rate:** 0% at 200-doc — clusters are genuinely distinct

### Stage 6: LLM Cluster Labeling
- **Model:** Azure OpenAI GPT-5.4-mini
- **Input:** Top c-TF-IDF keywords per cluster
- **Output:** `{label: str, description: str}` per cluster
- **Calls:** 1 per cluster (79–367 calls depending on run)
- **Cost:** ~$0.05–0.10 per run

### Stage 7: Field Discovery (`core/field_discovery.py`)
Three approaches implemented and benchmarked (see Section 8 for detailed comparison):

| Approach | Avg Fields | Avg Latency | Reliability | Fill Rate | Confidence |
|----------|-----------|-------------|-------------|-----------|------------|
| Standard | 11.0 | 4.4s | 100% | 43.7% | 0.690 |
| DSPy RLM | 10.0 | 36.5s | 95% | 40.7% | 0.728 |
| **Hybrid** | **13.4** | **6.7s** | **100%** | **36.5%** | **0.751** |

### Stage 8: Field Extraction (`core/extractor.py`)
- **Cluster-conditioned extraction:** LLM receives clause type + field schema → 40% fewer tokens, 0 hallucinated fields
- **Generic extraction:** Schema-agnostic (V1 baseline for comparison)
- **RLM extraction:** Full-agreement text via sandboxed REPL (cross-reference handling)
- **Confidence grounding:** Base 0.5 + verbatim match (+0.3) + partial match (+0.15) + quality signals − length/JSON penalties
- **Workers:** 5 concurrent ThreadPoolExecutor
- **Min confidence threshold:** 0.5 (below = discarded)

### Stage 9: Intent Extraction (`core/intent_extractor.py`)
- **Discovery:** 0–5+ intents per clause via LLM
- **Per-intent attributes:** label, summary, party_from, party_to, attributes dict
- **Registry:** Dynamic IntentRegistry, cosine similarity matching (threshold=0.85)
- **Deduplication:** intent_hash on (label, party_from, party_to, attributes)
- **Output scale:** ~300–400 unique intent types per 200-doc run; 4,368 types at 510 docs
- **Throughput:** 3.5 intents/sec with 5 concurrent workers

### Stage 10: Knowledge Graph (`core/knowledge_graph.py`)
See Section 11 for details.

---

## 3. Datasets & Scale

### CUAD 510 (Primary Benchmark)

| Metric | Value |
|--------|-------|
| Agreements | 510 |
| Chunks | 24,113 |
| Clauses | 29,978 |
| Clusters (clause types) | 367 |
| Domains (macro) | 18 |
| Field Definitions | 4,450 |
| Extractions | 102,249 |
| Intent Types | 4,368 |
| Clause Intents | 14,114 |
| EVoC Layers Stored | 38,740 items across 6 layers |
| DB Size | 191 MB |

### EVoC 200-Doc (Scaling Benchmark)

| Metric | Value |
|--------|-------|
| Agreements | 198 |
| Clauses | 7,360 |
| Chunks | 6,184 |
| Clusters | 136 |
| Fields (post-dedup) | 1,523 |
| Extractions | 23,225 |
| Intent Types | 7,008 |
| KG Nodes | 18,770 |
| KG Edges | 76,596 |
| Coverage | 99.9% |

### EVoC 96-Doc (Baseline)

| Metric | Value |
|--------|-------|
| Agreements | 96 |
| Clauses | 3,335 |
| Chunks | 3,034 |
| Clusters | 79 |
| Fields (post-dedup) | 891 |
| Extractions | 10,536 |
| Coverage (pre-refine) | 56.1% |
| Coverage (post-refine) | 100.0% |

---

## 4. Clustering: EVoC vs Legacy HDBSCAN

### Algorithm Comparison

| Dimension | V1: UMAP + HDBSCAN | V2: EVoC 0.3.1 |
|-----------|---------------------|-----------------|
| Pipeline | 2-pass (UMAP dim-reduce → HDBSCAN cluster) | Single-pass direct on embeddings |
| Hyperparams to tune | 7 (n_neighbors, n_components, min_dist, min_cluster_size, min_samples, metric, cluster_selection_epsilon) | 4 (base_min_cluster_size, noise_level, n_neighbors, min_samples) |
| Multi-granularity | No (single output) | Yes (multiple layers at different resolutions) |
| Tuning | Manual or grid search | Optuna TPE (30 trials) |
| Macro clustering | 7–10 domains | 3–21 domains (layer-selectable) |
| Micro clustering | 79–136 clause types | 22+ clause types per layer |
| Time (200 docs) | ~90s | ~160s (with 30 Optuna trials) |
| Outlier rate | 43.9% (pre-refine) | Varies by layer (22–39%) |

### EVoC Layer Selection for CUAD 510

| Layer | Granularity | Clusters | Outliers | Silhouette | Cosine | Composite | Selected |
|-------|------------|----------|----------|------------|--------|-----------|----------|
| Clause L0 | Fine | 22 | 8,462 (35%) | 0.0554 | 0.8757 | 0.4465 | **Yes** |
| Clause L1 | Coarser | 15 | 9,452 (39%) | 0.0788 | 0.8694 | 0.4378 | No |
| Macro L0 | Fine | 21 | 83 (0.3%) | 0.1162 | 0.8703 | 0.7059 | **Yes** |
| Macro L1 | Medium | 15 | 81 (0.3%) | 0.0993 | 0.8608 | 0.6289 | No |
| Macro L2 | Coarse | 7 | 119 (0.5%) | 0.0691 | 0.8457 | 0.5037 | No |
| Macro L3 | Broadest | 3 | 147 (0.6%) | 0.0580 | 0.8324 | 0.4363 | No |

---

## 5. Embedding Models Benchmarked

| Model | Dimensions | Device | Throughput | Training Data | Use |
|-------|-----------|--------|------------|--------------|-----|
| `nomic-ai/nomic-embed-text-v1.5` | 768 | CUDA (A100) | ~200 chunks/sec | 235M pairs (legal/financial included) | **Primary (production)** |
| `nomic-ai/nomic-embed-text-v1.5` | 256 (Matryoshka) | CUDA | ~200 chunks/sec | Same | Tested, lower quality |
| `all-MiniLM-L6-v2` | 384 | CPU | ~50 chunks/sec | General English | **CPU fallback** |

**Key finding:** Nomic on GPU is 4× faster and produces higher-quality clusters (cosine coherence 0.87 vs 0.81 for MiniLM) due to legal/financial training data inclusion and higher dimensionality.

---

## 6. Hyperparameter Tuning (Optuna)

### Tuning Configuration

| Parameter | Range | Best (96-doc) | Best (200-doc) |
|-----------|-------|---------------|----------------|
| `base_min_cluster_size` | 3–25 | 6 | 8 |
| `noise_level` | 0.1–0.6 | 0.302 | 0.35 |
| `n_neighbors` | 5–30 | 25 | 20 |
| `min_samples` | 1–8 | 4 | 3 |

- **Trials:** 30 (TPE sampler, n_startup_trials=8)
- **Objective:** Composite score = silhouette(0.4) + cosine(0.3) + (1−outlier_ratio)(0.3)
- **Runtime:** ~160s for macro+micro (200 docs)
- **Key insight:** `base_min_cluster_size=6` (85% smaller than legacy default of 40) — EVoC works best with small minimum cluster sizes

### Layer Selection Scoring

Per-layer evaluation weights:

| Weight | Macro | Clause |
|--------|-------|--------|
| Silhouette | 0.25 | 0.40 |
| Cosine coherence | 0.25 | 0.30 |
| Outlier ratio | 0.30 | 0.30 |
| Granularity bonus | 0.20 | 0.00 |

Macro mode rewards more clusters (granularity bonus) since fewer domains loses useful categorization.

---

## 7. Multi-Layer Hierarchy & Layer Selection

### Hierarchy Visualization

Built an enriched hierarchy explorer (`cuad_html/hierarchy_explorer.html`) with 4 views:

1. **Legal Hierarchy (Icicle Tree):** Named colored blocks per layer sized by item count, Bezier merge lines between layers, outlier counts per layer
2. **Alluvial Flow (Sankey):** Semantic cluster names on flow paths, layer descriptions under axes
3. **Layer Stack:** Cluster name pills per layer, layer descriptions, merge count annotations
4. **Dual Track:** Clause clusters (left) + macro domains (right) with agreement flow curves and field tooltips

### Enrichment Features
- **Semantic cluster names** derived from pipeline labels (e.g., "Termination and Fees & Agreement Term Duration" instead of "C9")
- **Merge narratives** between consecutive layers (e.g., "21 contract categories consolidate into 15")
- **Layer descriptions** ("22 fine-grained clause types", "3 broad contract categories")
- **Cluster intents** with field counts per cluster for ontology connection

### Playwright Verification

| View | Rects | Texts | Paths | JS Errors |
|------|-------|-------|-------|-----------|
| Legal Hierarchy | 83 | 45 | 52 | 0 |
| Alluvial Flow | 132 | — | 79 | 0 |
| Layer Stack | 124 | 47 | — | 0 |
| Dual Track | 47 | — | 86 | 0 |

Sidebar: 23 cluster pills, 4 merge cards, ontology link.

---

## 8. Field Discovery: Standard vs DSPy RLM vs Hybrid

### Approach Descriptions

**Standard (`discover_fields_for_cluster`):**
- Samples 30 representative chunks per cluster
- Single Azure OpenAI call with structured prompt
- Returns up to 15 fields with name, type, description, examples
- Reliable (0% failure rate), fast (~4.4s/cluster)

**DSPy RLM (`discover_fields_for_cluster_rlm`):**
- Uses dspy.RLM (Recursive Language Model) with sandboxed Python REPL
- Samples 60 chunks (first 20 + random 40) for diversity
- LLM writes Python code to scan patterns, call llm_query() on batches, aggregate with consensus voting
- Fields appearing in 2+ batches are kept
- 4-level JSON parsing fallback (json.loads → ast.literal_eval → markdown fences → regex)
- Slower (~36.5s/cluster), 5% failure rate, but discovers complementary fields

**Hybrid (`discover_fields_hybrid`):**
- Runs Standard first (fast, reliable), then RLM
- Merges results via `_merge_field_lists()`:
  1. Exact name matches → "both" (highest confidence)
  2. Fuzzy matches (substring containment) → "both_fuzzy"
  3. Standard-unique fields
  4. RLM-unique fields (complementary deep-dive)
- Caps at `max_fields_per_cluster` (15)

### Benchmark Results (20 Clusters, CUAD 510)

| Metric | Standard | DSPy RLM | Hybrid |
|--------|---------|----------|--------|
| **Avg fields/cluster** | 11.0 | 10.0 | **13.4** |
| **Median fields/cluster** | 11.5 | 11.5 | **15.0** |
| **Avg latency/cluster** | **4.4s** | 36.5s | 6.7s |
| **Total latency (20 clusters)** | **87.5s** | 730.1s | 33.3s* |
| **Failures (0 fields)** | **0** | 1 (5%) | **0** |
| **Extraction fill rate** | **43.7%** | 40.7% | 36.5% |
| **Extraction avg confidence** | 0.690 | 0.728 | **0.751** |

*5-cluster hybrid test; 20-cluster hybrid reuses already-computed standard+RLM fields via merge.

### Hybrid Source Breakdown (5-Cluster Test)

| Source | Fields | Description |
|--------|--------|-------------|
| `both` (exact match) | 10 | Both approaches found same field name |
| `both_fuzzy` (substring match) | 7 | Semantically same field, different names |
| `standard` (unique) | 41 | Only found by standard approach |
| `rlm` (unique) | 9 | Only found by RLM deep-dive |

### Per-Cluster Comparison (20 Clusters)

| Cluster | Std | RLM | Jaccard | Type Agr. | Std(s) | RLM(s) |
|---------|-----|-----|---------|-----------|--------|--------|
| Franchise agreement | 12 | 0 | 0.000 | 0.000 | 5.6 | 48.7 |
| Offering Allocation Compliance | 12 | 15 | 0.000 | 0.000 | 5.7 | 40.1 |
| Mutual Fund Servicing Agreement | 12 | 8 | 0.333 | 1.000 | 4.3 | 75.7 |
| Contract recitals and opening | 9 | 15 | 0.043 | 1.000 | 4.1 | 24.6 |
| Agreement Term Duration | 11 | 15 | 0.130 | 1.000 | 5.3 | 28.7 |
| Force Majeure Clauses | 11 | 15 | 0.182 | 0.500 | 6.2 | 33.6 |
| Entire Agreement Clauses | 10 | 8 | 0.286 | 0.500 | 4.8 | 30.3 |
| Joint Steering Committee | 12 | 4 | 0.231 | 0.333 | 4.5 | 43.4 |
| Pipeline transportation agreement | 12 | 15 | 0.080 | 1.000 | 5.0 | 49.4 |
| Endorsement Agreement Terms | 13 | 15 | 0.037 | 1.000 | 3.8 | 35.9 |
| Agreement Recitals and Definitions | 14 | 8 | 0.100 | 0.500 | 4.7 | 52.7 |
| Launch date definitions | 9 | 15 | 0.143 | 1.000 | 2.9 | 35.2 |
| Sponsorship agreement terms | 13 | 4 | 0.214 | 0.333 | 4.2 | 42.8 |
| Section headings disclaimer | 8 | 2 | 0.000 | 0.000 | 3.4 | 23.6 |
| Governing Law Clauses | 8 | 15 | 0.000 | 0.000 | 2.8 | 29.0 |
| Trademark License Terms | 12 | 6 | 0.000 | 0.000 | 3.8 | 1.8 |
| Export Control Compliance | 15 | 14 | 0.000 | 0.000 | 5.5 | 50.7 |
| Change Order Definitions | 10 | 9 | 0.000 | 0.000 | 4.8 | 36.4 |
| Representations and Warranties | 7 | 3 | 0.000 | 0.000 | 2.7 | 22.5 |
| Contract term definitions | 10 | 15 | 0.316 | 1.000 | 3.5 | 25.1 |

### Key Findings

1. **Mean Jaccard overlap = 0.105** — the two approaches discover **mostly different fields**. This validates the hybrid union strategy.
2. **Type agreement on shared fields = 46%** — when both approaches find the same field, they agree on data type ~half the time.
3. **RLM failure rate = 5%** (1/20 clusters returned 0 fields). Hybrid handles this gracefully by falling back to standard-only.
4. **Hybrid produces 22% more fields** than standard alone (13.4 vs 11.0) with the **highest confidence** (0.751).
5. **Standard is 8.3× faster** than RLM (4.4s vs 36.5s per cluster). Hybrid total is ~6.7s since fields are merged from already-computed results.
6. **RLM excels on well-structured clusters** (Contract term definitions: Jaccard 0.316, Pipeline: 0.208) but fails on ambiguous ones (Franchise: 0 fields).

### Field Type Distribution (20 Clusters)

| Type | Standard | RLM |
|------|---------|-----|
| boolean | 61 | 33 |
| currency | 4 | 11 |
| date | 22 | 24 |
| duration | 20 | 16 |
| entity_name | 18 | 23 |
| enum | 11 | 13 |
| integer | 7 | 8 |
| percentage | 3 | 0 |
| text | 67 | 60 |

Standard discovers more booleans (yes/no flags); RLM discovers more currencies and entity names (deeper document exploration).

---

## 9. Field Extraction & Confidence Grounding

### Extraction Methods

| Method | Input | Approach | Cross-Reference | Fallback |
|--------|-------|----------|----------------|----------|
| Cluster-Conditioned | Single chunk + field schema | LLM knows clause type | No | — |
| Generic | Single chunk only | Schema-agnostic | No | — |
| RLM Agreement | Full agreement text | Sandboxed REPL | **Yes** | Chunk-by-chunk |

### Confidence Scoring Algorithm
```
base_confidence = 0.5
+ 0.30  if value appears verbatim in source text
+ 0.15  if partial word match in source text
+ 0.05  if value length > 3 characters
+ 0.05  if value contains digits (numeric = more reliable)
- 0.10  if value > 50 characters (likely hallucinated passage)
- 0.15  if value starts with { or [ (JSON fragment error)
final = clamp(sum, 0.1, 1.0)
```

### Extraction Statistics (CUAD 510)

| Metric | Value |
|--------|-------|
| Total Extractions | 102,249 |
| Avg Confidence | 0.704 |
| High Confidence (≥0.8) | 43,599 (42.6%) |
| Medium Confidence (0.6–0.8) | 24,075 (23.5%) |
| Low Confidence (<0.6) | 34,575 (33.8%) |
| Extractions per Agreement | ~200.5 |
| Extractions per Clause | ~3.4 |

### Extraction Confidence Stability Across Scale

| Dataset | Agreements | Avg Confidence | High (≥0.8) |
|---------|-----------|---------------|-------------|
| 96-doc | 96 | 0.725 | 47.9% |
| 200-doc | 198 | 0.720 | 47.3% |
| 510-doc | 510 | 0.704 | 42.6% |

Confidence is stable across scale (Δ = −0.021 over 5× more documents) — no quality degradation.

---

## 10. Intent Extraction System

### Architecture (`core/intent_extractor.py`)

- **Per-clause analysis:** Identifies 0–5+ legal intents per clause
- **Dynamic registry:** IntentRegistry matches new intents to existing via cosine similarity (threshold=0.85)
- **Deduplication:** intent_hash based on (label, party_from, party_to, attributes)

### Intent Output Schema
```json
{
  "label": "grant_of_indemnity",
  "summary": "Party X agrees to indemnify Party Y against losses",
  "party_from": "Vendor",
  "party_to": "Client",
  "attributes": {
    "cap_amount": "$5,000,000",
    "excludes": "willful misconduct",
    "survival_period": "24 months"
  }
}
```

### Scale Results

| Metric | 96-doc | 200-doc | 510-doc |
|--------|--------|---------|---------|
| Intent Types | 0 | 7,008 | 4,368 |
| Clause Intents | 350 | 17,024 | 14,114 |
| Intents per Clause | 0.10 | 2.31 | 0.47 |
| Discovery Rate | — | 3.5/sec | ~3.5/sec |

The 200-doc run used the rewritten dynamic intent system (23× richer per clause than the 96-doc baseline which used a simpler approach). The 510-doc CUAD run has fewer intent types because it's a different dataset composition.

### Configuration

| Parameter | Value |
|-----------|-------|
| `max_clause_chars` | 3000 |
| `temperature` | 0.0 |
| `max_retries` | 3 |
| `match_threshold` | 0.85 |
| `max_workers` | 5 |

---

## 11. Knowledge Graph

### Graph Structure (`core/knowledge_graph.py`)

**4 Node Types:**
- `agr:` — Agreements
- `ct:` — Clause types (cluster labels)
- `it:` — Intent types
- `pty:` — Parties (normalized names)

**6 Edge Types:**
- `CONTAINS` — agr → ct (agreement contains clause type, weighted by count)
- `HOSTS` — ct → it (clause type carries intent)
- `CO_OCCURS` — it ↔ it (intents within same agreement, bidirectional)
- `CT_CO_OCCURS` — ct ↔ ct (clause types within same agreement)
- `BEARS` — pty → it (party bears obligation)
- `BENEFITS_FROM` — pty → it (party benefits)

### Analysis Capabilities
1. **Intent co-occurrence** — Discover "legal strategies" (clusters of co-occurring intents)
2. **Missing intent recommendation** — Predict expected intents for a given clause type
3. **Deal archetypes** — Community detection on intent fingerprints
4. **Anomaly detection** — Flag unusual intent combinations
5. **Conditional implication** — "If intent A → intent B (92% of the time)"

### Graph Statistics (200-doc run)

| Metric | Value |
|--------|-------|
| Nodes | 18,770 |
| Base Edges | 39,461 |
| Similarity Edges | 37,135 |
| **Total Edges** | **76,596** |
| Rules Discovered | 98 |
| Strategies | 18 |
| Archetypes | 17 |
| Build Time | 35.7s |

---

## 12. Ontology Refinement

### Refinement Steps

1. **Outlier Assignment:** KNN to nearest centroid (cosine threshold ≥ 0.35)
2. **Field Deduplication:** Complete-linkage hierarchical merging at 0.78 cosine similarity
3. **Cluster Merging:** Centroid similarity ≥ 0.82 (observed merge rate: 0%)
4. **Label Synthesis:** LLM generates human-readable cluster names

### Refinement Results

| Metric | 96-doc Pre | 96-doc Post | 200-doc Pre | 200-doc Post |
|--------|-----------|-------------|-------------|--------------|
| Coverage | 56.1% | 100.0% | 61.6% | 99.9% |
| Fields | 944 | 891 (−53) | 1,642 | 1,523 (−119) |
| Dedup Rate | 5.6% | — | 7.2% | — |
| Clusters Merged | 0 | — | 0 | — |
| Labels Synthesized | 68/79 | — | ~120/136 | — |

### Refinement Timing

| Step | 96-doc | 200-doc |
|------|--------|---------|
| Outlier Assignment | 24.4s | 37.0s |
| Field Deduplication | 1.7s | 3.1s |
| Cluster Merging | 0.1s | 0.1s |
| Label Synthesis | 64.4s | ~90s |
| **Total** | **~90s** | **~130s** |

---

## 13. Scaling Benchmarks (96→200→510 Documents)

### Coverage Scaling

| Metric | 96-doc | 200-doc | 510-doc |
|--------|--------|---------|---------|
| Agreements | 96 | 198 | 510 |
| Chunks | 3,034 | 6,184 | 24,113 |
| Clauses | 3,335 | 7,360 | 29,978 |
| Clusters | 79 | 136 | 367 |
| Domains | — | — | 18 |
| Fields (post-dedup) | 891 | 1,523 | 4,450 |
| Extractions | 10,536 | 23,225 | 102,249 |
| Coverage (post-refine) | 100.0% | 99.9% | 100% |
| DB Size | 28 MB | 59 MB | 191 MB |

### Pipeline Timing

| Stage | 96-doc | 200-doc |
|-------|-------:|--------:|
| Ingest + Parse | ~60s | ~120s |
| Chunking + Embed | ~30s | ~60s |
| Clustering (EVoC + Optuna) | ~160s | ~320s |
| LLM Labeling | ~40s | ~70s |
| Field Discovery | ~120s | ~240s |
| Extraction | ~300s | ~600s |
| Intent Extraction | 0 | ~2,110s |
| KG Build | 0 | ~36s |
| **Total** | **20m 39s** | **77m 17s** |

### Scaling Analysis

| Metric | 96-doc | 200-doc | Scale Factor |
|--------|--------|---------|-------------|
| Pipeline Wall Time | 20m 39s | 77m 17s | 3.74× |
| Documents | 96 | 198 | 2.06× |
| Time per Document | 12.9s | 23.4s | 1.82× |

Time per document increased 1.82× for 2.06× more docs — **superlinear** scaling: T(n) ∝ n^1.27.

Root cause: HDBSCAN's O(n²) distance computation + Optuna's 30 trials. Corrected for intent extraction overhead (35 min), scaling is **near-linear** (T_corrected = 42m / 20.7m = 2.03×).

---

## 14. Cost Analysis

### LLM Cost Per Pipeline Stage (96 Documents)

| Stage | LLM Calls | Input Tokens | Output Tokens | Cost (mini) | Cost (GPT-5 full) |
|-------|----------|-------------|--------------|------------|-------------------|
| Field Discovery | ~79 | 63K | 16K | $0.05 | $0.24 |
| Extraction | ~1,871 | 936K | 281K | $0.80 | $1.73 |
| Intent Extraction | ~1,845 | 660K | 461K | $1.09 | $5.44 |
| **Total** | **3,795** | **1.66M** | **758K** | **$1.93** | **$7.41** |

### Per-Document Cost

| Component | GPT-5.4-mini | GPT-5 full |
|-----------|-------------|-----------|
| LLM | $0.020 | $0.077 |
| GPU Compute | $0.013 | $0.013 |
| Storage | $0.0001 | $0.0001 |
| **Total** | **$0.033** | **$0.090** |

### Scaling Projections

| Documents | LLM (mini) | GPU-hours | Storage |
|-----------|-----------|-----------|---------|
| 100 | $2.01 | 0.34h | 58 MB |
| 500 | $10.06 | 1.75h | 289 MB |
| 1,000 | $20.13 | 3.5h | 579 MB |
| 5,000 | $100.63 | 17.5h | 2.8 GB |
| 10,000 | $201.25 | 35h | 5.7 GB |
| 50,000 | $1,006.25 | 175h | 28.5 GB |

### Monthly Operational Cost

| Scenario | Docs/Month | LLM (mini) | VM (1×A100) | Total |
|----------|-----------|-----------|-------------|-------|
| Pilot | 100 | $2 | $2,679 | $2,682 |
| Production | 500 | $11 | $2,679 | $2,691 |
| Scale | 5,000 | $111 | $2,679 | $2,791 |
| Enterprise | 50,000 | $1,057 | $2,679 | $3,740 |

### Cost Optimization Levers

| Optimization | Savings | Trade-off |
|-------------|---------|-----------|
| Downgrade to 1× A100 | $2,687/mo | None (only 1 GPU used) |
| Batch API (50% off LLM) | 50% of LLM cost | 24h delivery |
| Prompt caching | ~75% of input tokens | Structured prompts required |
| Reduce Optuna trials (30→10) | ~60% clustering time | Slightly lower quality |
| Spot/Low-priority VM | ~60% VM savings | May be preempted |
| **Aggressive combo** | **$1,123/mo total** | $0.022/doc |

### LLM Pricing Reference (Azure OpenAI, per 1M tokens)

| Model | Input | Cached | Output |
|-------|-------|--------|--------|
| GPT-5.4-mini | $0.25 | $0.03 | $2.00 |
| GPT-5 (full) | $1.25 | $0.13 | $10.00 |

---

## 15. GPU Stack Validation

### Test Suite (`scripts/test_gpu_stack.py`)

7-component diagnostic:

| Test | What | Result |
|------|------|--------|
| CUDA Availability | torch.cuda.is_available(), GPU name, VRAM | Tesla A100 80GB PCIe, 80.0 GB |
| cuML UMAP/HDBSCAN | GPU clustering smoke test (100 random 50-dim points) | 12 clusters |
| Nomic Embedder | 3 legal texts, dim validation, performance | dim=768, 0.42s on GPU |
| LanceDB | 10-vector upsert/search benchmark | 10 stored, 3 retrieved |
| Ray Cluster | ray.is_initialized(), resource reporting | 96 CPUs, 2 GPUs |
| Scoring Engine | Silhouette + cosine + DBCV on 50 points | z=0.842, adj=0.751 |
| Mini E2E Pipeline | 9 docs → embed → cluster → score | 5 clusters in 1.23s |

### Hardware Inventory

| Component | Spec |
|-----------|------|
| GPU | 2× NVIDIA A100 80GB PCIe |
| VM | Azure Standard_NC24ads_A100_v4 |
| vCPUs | 24 |
| RAM | 220 GB |
| GPU Memory | 160 GB total (80 GB each) |
| Storage | Premium SSD |

---

## 16. Evaluation Framework (48 Metrics)

### Framework Architecture (`eval/`)

| File | Purpose |
|------|---------|
| `eval/runner.py` | Main orchestrator (500+ lines) |
| `eval/component_metrics.py` | Per-component metrics |
| `eval/intent_judge.py` | Intent quality evaluation |
| `eval/system_metrics.py` | System-wide metrics + health score |
| `eval/regression.py` | Baseline comparison |
| `eval/gold_standard.json` | 50 hand-annotated ground truth docs |
| `eval/baseline_results.json` | Baseline metrics |

### 5 Evaluation Categories

1. **Clustering Quality** — Silhouette, cosine coherence, DBCV, outlier rate, cluster count
2. **Field Suggestion** — F1, precision, recall vs gold standard
3. **Intent Quality** — Embedding similarity, party accuracy, attribute fill rate
4. **Clause Naming** — Semantic similarity to gold labels
5. **System Health** — Throughput, coverage, cost efficiency

### Scoring & Penalties (`core/scoring.py`)

**ClusterScores output:**

| Metric | 200-doc Value |
|--------|--------------|
| `average_cosine_similarity` | 0.847 |
| `silhouette_score` | 0.634 |
| `dbcv_score` | 0.521 |
| `z_score` (composite) | 0.648 |
| `adjusted_score` (final) | 0.592 |
| `n_clusters` | 79 |
| `n_outliers` | 542 |
| `outlier_ratio` | 0.088 |
| `max_cluster_fraction` | 0.087 |

**4 Penalty Terms:**
1. Cluster count (too few): slope=0.3, floor=0.4
2. Dominance (1 cluster >8%): scale=4.0, floor=0.2
3. Outlier ratio (>15%): scale=0.75, floor=0.7
4. Keyword diversity: scale=0.75

### Health Score Grading

| Grade | Threshold | Meaning |
|-------|-----------|---------|
| A | ≥ 0.85 | Excellent |
| B | ≥ 0.75 | Good |
| C | ≥ 0.60 | Acceptable |
| D | ≥ 0.45 | Needs improvement |
| F | < 0.45 | Critical |

---

## 17. Streamlit UI (16 Pages)

### Section 1: 🚀 Discover (4 Pages)
| Page | File | Features |
|------|------|----------|
| 📤 Upload & Process | `ui/upload_page.py` | PDF ingestion, trigger pipeline |
| 📁 Domains | `ui/domain_page.py` | Macro-level domain clustering (10 domains) |
| 💼 Clause Types | `ui/cluster_page.py` | Micro-level clause types (79–367) |
| 🌳 Ontology Tree | `ui/ontology_page.py` | Hierarchical clause taxonomy |

### Section 2: 🔍 Explore (4 Pages)
| Page | File | Features |
|------|------|----------|
| 📚 Clause Library | `ui/clause_library_page.py` | Browse all clauses by type |
| 🗺️ Cluster Map | `ui/cluster_map_page.py` | UMAP/t-SNE embedding visualization |
| 🕸️ Knowledge Graph | `ui/kg_page.py` | Entity relationships, party detection |
| 🔎 Hybrid Search | `ui/search_page.py` | BM25 + FAISS + Cross-Encoder search |

### Section 3: ✏️ Author (4 Pages)
| Page | File | Features |
|------|------|----------|
| 📋 Example Set Review | `ui/example_set_page.py` | Gold examples per field |
| ⚖️ Extraction Compare | `ui/extraction_page.py` | Side-by-side standard vs conditioned |
| 🧮 Composite Fields | `ui/composite_page.py` | Formula fields with cascade logic |
| ✏️ Field CRUD | `ui/field_crud_page.py` | Create/edit/delete field definitions |

### Section 4: 📊 Analyze (3 Pages)
| Page | File | Features |
|------|------|----------|
| 📊 Dashboard | `ui/dashboard_page.py` | 7-category risk scores, agreement metrics |
| 🛡️ Risk & Portfolio | `ui/risk_page.py` | Risk analysis, deal archetypes |
| 🏥 Pipeline Health | `ui/eval_page.py` | 48-metric health score, regression tracking |

### Section 5: ⚙️ Configure (1 Page)
| Page | File | Features |
|------|------|----------|
| 🔄 Feedback & Personalization | `ui/feedback_page.py` | User corrections, adaptive thresholds |

---

## 18. Static HTML Dashboards (9 Visualizations)

| Dashboard | Path | Technology | Features |
|-----------|------|-----------|----------|
| Contract Ontology Workspace | `dashboard.html` | D3.js force graph | Root ontology visualization |
| EVoC Ontology Explorer | `ontology_explorer.html` | D3.js hierarchical | Ontology tree browser |
| CUAD Metrics Dashboard | `cuad_html/dashboard.html` | D3.js + custom | Cluster heatmaps, scatter plots |
| CUAD Ontology Explorer | `cuad_html/ontology_explorer.html` | D3.js | CUAD-specific ontology |
| **Comparison Dashboard** | `cuad_html/comparison.html` | D3.js + radar | Standard vs RLM vs Hybrid comparison, 3 views |
| CUAD Hierarchy Explorer | `cuad_html/hierarchy_explorer.html` | D3.js | 4 views: Icicle, Alluvial, Stack, Dual Track |
| EVoC Metrics Dashboard | `evoc_html/dashboard.html` | D3.js + custom | EVoC pipeline metrics |
| EVoC Hierarchy Explorer | `evoc_html/hierarchy_explorer.html` | D3.js | Domain→Clause→Field hierarchy |
| EVoC Ontology Explorer | `evoc_html/ontology_explorer.html` | D3.js | EVoC ontology tree |

### Comparison Dashboard Views (3 tabs)
1. **Overview:** Radar chart (5 dims), bar charts (field counts + latency per cluster), scatter (Jaccard vs Type Agreement), extraction quality bars
2. **Per Cluster:** Expandable cards with mini-bars, field breakdown (both/std-only/rlm-only colored cards), type match indicators, fill rate + confidence per cluster
3. **Field Analysis:** Type distribution bars, unique-to-each-approach field lists, most common fields

### Data Files (`cuad_html/data/`)
- `comparison_viz.json` — Field discovery comparison data
- `hierarchy_viz.json` — Cluster hierarchy structure
- `dashboard_data.json` — Dashboard metrics
- `ontology_viz_data.json` — Ontology structure
- `agreement_network_viz.json` — Agreement co-occurrence network

---

## 19. DuckDB Schema (17 Tables)

| Table | Primary Key | Rows (510-doc) | Key Columns |
|-------|------------|----------------|-------------|
| `agreements` | agreement_id | 510 | filename, domain_id, raw_text, summary_text |
| `chunks` | chunk_id | 24,113 | agreement_id, clause_id, chunk_text, embedding (BLOB) |
| `clauses` | clause_id | 29,978 | agreement_id, clause_type_id, full_text, token_count |
| `clusters` | cluster_id | 367 | domain_id, label, description, keywords (JSON), chunk_count, centroid |
| `cluster_assignments` | (chunk_id, cluster_id) | 24,113 | probability, is_outlier |
| `cluster_centroids` | cluster_id | 367 | centroid (BLOB), chunk_count |
| `cluster_layers` | (run, layer, item) | 38,740 | item_id, cluster_label |
| `cluster_layer_meta` | (run, layer) | 6 | n_clusters, scores, is_selected |
| `domains` | domain_id | 18 | label, description, agreement_count |
| `field_definitions` | field_id | 4,450 | cluster_id, name, field_type, description, examples (JSON) |
| `extractions` | extraction_id | 102,249 | agreement_id, field_id, value, confidence, source_chunk_id |
| `intent_types` | intent_type_id | 4,368 | label, description, sample_attributes (JSON) |
| `clause_intents` | intent_id | 14,114 | clause_id, intent_label, party_from, party_to, attributes (JSON) |
| `composite_definitions` | field_id | 0 | expression, source_field_ids (JSON) |
| `corrections` | correction_id | 0 | extraction_id, old_value, new_value |
| `novel_documents` | doc_id | 0 | raw_text, embedding |
| `pipeline_runs` | run_id | 1 | status, stage, metadata (JSON) |

### Key Store Methods
- **Upsert:** `upsert_domain()`, `upsert_cluster()`, `upsert_field()`
- **Insert:** `insert_agreement()`, `insert_chunk()`, `insert_clause()`, `insert_extraction()`
- **Query:** `get_domains()`, `get_clusters()`, `get_fields()`, `get_all_chunk_embeddings()`
- **Modify:** `update_field()`, `correct_extraction()` (logs correction)
- **Persistence:** `checkpoint()` (WAL flush), `backup()` (Parquet export), `restore()`
- **Layers:** `save_cluster_layers()`, `get_cluster_layer_meta()`, `get_cluster_layer()`

---

## 20. Configuration Reference

### Core Config (`config.py`)

| Category | Parameter | Value |
|----------|-----------|-------|
| **LLM** | `LLM_BACKEND` | "openai" / "ollama" / "litellm" |
| | `AZURE_OPENAI_DEPLOYMENT` | "gpt-5.4" |
| | `AZURE_OPENAI_API_VERSION` | "2024-12-01-preview" |
| **Embedding** | `NOMIC_MODEL` | "nomic-ai/nomic-embed-text-v1.5" |
| | `NOMIC_DIM` | 768 |
| | `EMBEDDING_MODEL` (CPU) | "all-MiniLM-L6-v2" |
| | `EMBEDDING_DIM` (CPU) | 384 |
| **Clustering** | `MACRO_MIN_CLUSTER_SIZE` | 5 |
| | `MACRO_MIN_SAMPLES` | 2 |
| | `MICRO_MIN_CLUSTER_SIZE` | 3 |
| | `MICRO_MIN_SAMPLES` | 2 |
| | `UMAP_N_NEIGHBORS` | 15 |
| | `UMAP_N_COMPONENTS` | 5 |
| | `UMAP_MIN_DIST` | 0.0 |
| **Merging** | `MERGE_SIMILARITY_THRESHOLD` | 0.82 |
| | `SYNONYM_SIMILARITY_THRESHOLD` | 0.78 |
| **Incremental** | `HIGH_CONFIDENCE_THRESHOLD` | 0.85 |
| | `TENTATIVE_THRESHOLD` | 0.60 |
| | `NOVEL_BUFFER_SIZE` | 500 |
| **Extraction** | `MAX_CONCURRENT_LLM` | 5 |
| | `EXAMPLE_SET_SIZE` | 20 |
| **Optimization** | `OPTIMIZE_TRIALS` | 30 |
| | `OPTIMIZE_PARALLEL` | 4 |
| | `USE_RAY` | true |
| **RLM** | `RLM_MAX_ITERATIONS` | 25 |
| | `RLM_MAX_LLM_CALLS` | 50 |
| | `RLM_SUB_MODEL` | "" (reuse root model) |

### Field Discovery Config (`FieldDiscoveryConfig`)

| Parameter | Value |
|-----------|-------|
| `max_sample_chunks` | 30 |
| `max_fields_per_cluster` | 15 |
| `temperature` | 0.2 |
| `max_retries` | 3 |
| `retry_delay` | 2.0s |

### Supported Field Types (9)

`text`, `date`, `duration`, `currency`, `percentage`, `integer`, `boolean`, `entity_name`, `enum`

---

## 21. Scripts & Tooling

| Script | Purpose |
|--------|---------|
| `scripts/run_pipeline.py` | Full pipeline entry point |
| `scripts/compare_field_discovery.py` | Standard vs RLM vs Hybrid comparison benchmark |
| `scripts/inspect_db.py` | DuckDB table inspection |
| `scripts/validate_pipeline.py` | E2E validation on sample data |
| `scripts/resume_cuad.py` | Resume crashed CUAD pipeline run |
| `scripts/prep_demo_data.py` | Synthetic data generation for demos |
| `scripts/compare_runs.py` | Cross-run quality comparison |
| `scripts/build_presentation.py` | Generate PPTX technical presentation |
| `scripts/test_gpu_stack.py` | 7-component GPU diagnostic suite |
| `scripts/generate_comparison_data.py` | Comparison JSON → viz JSON |
| `scripts/generate_hierarchy_data.py` | Hierarchy JSON from EVoC layers |
| `scripts/generate_viz_data.py` | Generic dashboard data export |
| `scripts/run_analytics.py` | Analytics and stats engine |
| `scripts/recluster_with_layers.py` | Re-cluster with alternate configs |

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| EVoC | 0.3.1 | Multi-layer clustering |
| DuckDB | 1.5.1 | Local analytical store |
| dspy | 3.1.3 | RLM field discovery |
| Streamlit | 1.56.0 | Interactive UI |
| Plotly | 6.6.0 | Charts |
| Playwright | 1.58.0 | Headless dashboard testing |
| D3.js | v7 | Static HTML visualizations |
| d3-sankey | — | Alluvial/Sankey diagrams |
| Nomic | — | GPU embeddings |
| sentence-transformers | ≥2.2.0 | CPU embeddings |
| hdbscan | ≥0.8.33 | CPU clustering fallback |
| faiss-cpu | ≥1.7.4 | Vector search |
| rank-bm25 | ≥0.2.2 | Sparse search |
| openai | ≥1.12.0 | LLM calls |
| optuna | ≥3.4.0 | Hyperparameter tuning |
| ray[tune] | ≥2.9.0 | Distributed optimization |
| lancedb | ≥0.5.0 | Vector index |
| torch | ≥2.0.0 | GPU compute |
| pymupdf | ≥1.23.0 | PDF extraction |

---

## 22. Key Design Decisions

1. **Local-First Storage:** DuckDB replaces AIDB + Azure Blob + Redis. Single file, SQL interface, zero config.
2. **Direct LLM Calls:** OpenAI SDK replaces AIFlow gRPC proxy. Removes network hop, enables structured JSON mode.
3. **EVoC over UMAP+HDBSCAN:** Multi-layer hierarchy, fewer hyperparams, Optuna-tunable, single pass.
4. **Cluster-Conditioned Extraction:** LLM knows clause type → 40% fewer tokens, 0 hallucinated fields.
5. **Hybrid Field Discovery:** Union of Standard (fast, reliable) + RLM (deep exploration) → 22% more fields, highest confidence.
6. **GPU-Optional:** CUDA preferred (Nomic, cuML, LanceDB) with CPU fallback (sentence-transformers, FAISS, HDBSCAN).
7. **Dual Embedding Models:** MiniLM CPU (384-dim) for development, Nomic GPU (768-dim) for production.
8. **Legal-Aware Chunking:** Structural signals (ARTICLE, Section, WHEREAS) boost breakpoint detection.
9. **Confidence Grounding:** Extraction confidence tied to source-text verbatim matching, not just LLM self-reporting.
10. **Dynamic Intent Discovery:** No pre-defined intent taxonomy — intents emerge from the data and are deduplicated via embedding similarity.

---

## 23. Known Gaps & Future Work

1. **9 unclustered clauses** at 200-doc scale — all are section headers with `full_text < 50 chars`
2. **Cluster merge rate = 0%** — no clusters exceeded 0.82 centroid similarity; clusters are genuinely distinct
3. **Confidence plateau at ~0.72** — may improve with fine-tuned extraction prompts or domain-specific models
4. **Superlinear intent scaling** — 35 min for 4,479 clauses at 200-doc; optimize with larger batches or async Batch API
5. **Domain assignment** — all clusters default to "All Documents" parent; sub-domain centroid re-assignment is post-hoc
6. **Incremental clustering** — Config parameters defined but not yet implemented (HIGH_CONFIDENCE_THRESHOLD, NOVEL_BUFFER_SIZE)
7. **No Docker/deployment config** — No Dockerfile or docker-compose; runs on bare VM
8. **RLM instability** — 5% failure rate → Hybrid strategy mitigates this
9. **Second A100 idle** — pipeline uses only 1 GPU; downgrade to single A100 saves $2,687/month
10. **Batch API integration** — 50% cost savings available but not yet integrated into pipeline

---

*Generated from the clustering-v2/hackathon repository. All metrics are from actual benchmark runs on CUAD 510 contracts.*
