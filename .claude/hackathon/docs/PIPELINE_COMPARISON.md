# Pipeline Comparison: Macro-Micro vs Clause-Type vs Hybrid Clustering

> **Dataset**: 96–98 public legal agreements, ~3,000–3,159 chunks  
> **Hardware**: 2x NVIDIA A100 80GB, Azure OpenAI gpt-4o-mini  
> **Embedding model**: nomic-embed-text-v1.5 (768-dim, CUDA)

---

## 1. Architecture Overview

### Macro-Micro Pipeline (`core/cluster_pipeline.py`)

```
Documents -> Macro Cluster (by document embedding) -> 7 domains
     -> Per-domain Micro Cluster (by chunk embedding) -> 80 clusters
          -> Field Discovery (per cluster) -> 918 fields, 10,228 extractions
```

- **Stage 1 — Macro**: BERTopic/HDBSCAN on document-level embeddings -> 7 coarse domains
- **Stage 2 — Micro**: Per-domain BERTopic/HDBSCAN with Optuna tuning on chunk embeddings -> 80 clusters
- **Field discovery**: 15 fields max per cluster, 5 async workers, confidence filtering

### Clause-Type Pipeline (`core/clause_pipeline.py`)

```
Documents -> Clause Extraction (structural split) -> 2,002 valid clauses
     -> Clause Embedding (mean of chunk embeddings) -> Flat HDBSCAN -> 13 clause types
          -> Field Discovery (per clause type) -> 158 fields, 11,837 extractions
```

- **Single stage**: Computes clause-level embeddings (mean over constituent chunks), clusters ALL clauses corpus-wide in a single HDBSCAN pass
- **LLM labeling**: Azure OpenAI labels each clause type with descriptive name
- **Complete-linkage merge**: Near-duplicate types merged at 0.93 cosine threshold
- **Outlier assignment**: Unassigned clauses mapped to nearest type

### EVoC Hybrid Pipeline (`core/evoc_pipeline.py --mode hybrid`) — NEW

```
Documents -> Clause Extraction (structural split) -> 1,928 valid clauses
     -> Clause Embedding (mean of chunk embeddings) -> EVoC + Optuna -> 79 clause types
     -> Document Embedding (field-aware summaries) -> EVoC macro -> 10 domains
          -> Field Discovery (per clause type) -> 944 fields, 10,536 extractions
```

- **Clause clustering**: EVoC with 30-trial Optuna optimization on clause embeddings
- **Smart layer selection**: Scoring-based (silhouette + cosine + outlier ratio) instead of persistence-based
- **Document domains**: Second EVoC pass on summary embeddings for hierarchical organization
- **Dual output**: clause_types.json (types -> clauses -> fields) + domain assignments per document

---

## 2. Head-to-Head Metrics

| Metric | Macro-Micro | Clause-Type | **EVoC Hybrid** |
|---|---|---|---|
| **Algorithm** | UMAP + HDBSCAN + Optuna | UMAP + HDBSCAN | **EVoC + Optuna** |
| **Groups** | 7 domains -> 80 clusters | 13 clause types | **10 domains + 79 clause types** |
| **Total fields** | 918 | 158 | **944** |
| **Total extractions** | 10,228 | 11,837 | **10,536** |
| **Extractions/field** | 11.1 avg | 74.9 avg | **11.2 avg** |
| **Low-confidence filtered** | 0 | 269 | **215** |
| **Outlier handling** | 6 docs, 149 chunks (5.2%) | 366 clauses (18.3%) | **505 clauses (26.2%)** |
| **Avg quality score** | 0.908 | 0.883 | **0.920** |
| **Total time** | 33.4 min | 22.2 min | **20.7 min** |

### Stage Timing Breakdown

| Stage | Macro-Micro | Clause-Type | **EVoC Hybrid** |
|---|---|---|---|
| Parse | 0.5s | 0.5s | 0.4s |
| Chunk | 34.1s | 34.2s | 34.2s |
| Embed | 57.6s | 64.4s | 15.9s (cached) |
| Persist (DB + Lance) | 19.8s | 46.0s | ~46s |
| **Clustering** | **973.1s** (macro 8s + micro 965s) | **468.2s** | **~160s** (Optuna 90s + macro 10s) |
| **Field Discovery** | **917.0s** (80 clusters) | **717.6s** (13 clusters) | **976.5s** (79 types) |
| **Total** | **2,006s** | **1,334s** | **1,241s** |

EVoC Hybrid's clustering is **6x faster** than HDBSCAN M-M despite running Optuna, because EVoC eliminates the UMAP step. Total time is **38% faster** than HDBSCAN M-M.

---

## 3. Cluster Quality Comparison

### Macro-Micro: 7 Domains, 80 Clusters

| Domain | Clusters | Chunks | Avg Quality |
|---|---|---|---|
| Service Contractor Agreements | 16 | 853 | 0.908 |
| Real Estate Leases | 15 | 453 | 0.919 |
| Data Sharing Security Agreements | 14 | 397 | 0.894 |
| Loan and Guarantee Agreements | 14 | 299 | 0.918 |
| Government Procurement | 11 | 416 | 0.899 |
| Software and Hardware Sales | 7 | 151 | 0.907 |
| Director Equity Compensation | 3 | 325 | 0.911 |

### Clause-Type: 13 Types

| Clause Type | Clauses | Agreements | Fields | Quality |
|---|---|---|---|---|
| Memorandum of Understanding | 235 | 51 | 10 | 0.865 |
| Independent Contractor | 186 | 76 | 10 | 0.893 |
| Terms and Conditions of Sale | 113 | 10 | 14 | 0.886 |
| Loan and Guarantee Agreement | 89 | 4 | 12 | 0.891 |
| Definitions / Settlement Rate | 87 | 4 | 14 | 0.868 |
| Table of Contents / Definitions | 84 | 19 | 10 | 0.874 |
| Property Description | 77 | 2 | 12 | 0.867 |
| Lease Compliance | 66 | 8 | 12 | 0.873 |
| Special Pricing / Tariff Addendum | 65 | 1 | 14 | 0.991 |
| Consulting Services | 61 | 8 | 13 | 0.888 |
| Guaranty Agreement | 60 | 1 | 15 | 0.924 |
| Executive Change of Control | 59 | 6 | 12 | 0.895 |
| Insurance Requirements | 58 | 11 | 10 | 0.877 |

### EVoC Hybrid: 79 Clause Types + 10 Domains

The hybrid pipeline discovers **6x more clause types** than HDBSCAN (79 vs 13), providing much finer clause-level semantics. Sample types include:

- Change Order Price Adjustments, Data Sharing Agreement, Employee IP Assignment
- Confidentiality Obligations, Insurance Provisions, Notice Provisions
- Contract Amendment Terms, Severability Clauses, Signature Blocks
- Property Boundary Descriptions, Lease Amendment Documents
- Procurement Manual Sections, Research Services Agreement

The 10 document domains provide hierarchical navigation without sacrificing clause-level detail.

---

## 4. Pros and Cons

### Macro-Micro Pipeline

**Pros**
- Higher quality (0.908 avg) from Optuna-tuned per-domain clustering
- 80 fine-grained clusters capture narrow topic distinctions
- More total fields (918) covering long-tail patterns
- Lowest outlier rate (5.2%)
- Domain hierarchy for navigation

**Cons**
- **Slowest** at 33.4 min (Optuna per domain = 965s clustering)
- No clause awareness — clusters chunks, not clauses
- Significant field duplication (56 names in 2+ clusters)
- Documents clustered at wrong granularity (7 coarse domains)

### Clause-Type Pipeline

**Pros**
- Clause-level semantics — groups contractual clauses cross-document
- Highly reusable fields (74.9 extractions/field)
- More total extractions (11,837) with fewer fields
- 34% faster than macro-micro (22.2 min)

**Cons**
- 13 types too coarse — "MOU / Project Development" (235 clauses) is a catch-all
- Higher outlier rate (18.3%)
- No document-level organization
- Fewer fields (158) — may miss important clause patterns

### EVoC Hybrid Pipeline — RECOMMENDED

**Pros**
- **79 clause types** — 6x more granularity than HDBSCAN clause
- **10 document domains** — hierarchical organization (best of macro-micro)
- **944 fields** — matching macro-micro's comprehensive coverage
- **0.920 avg quality** — highest across all pipelines
- **38% faster** than macro-micro (20.7 min vs 33.4 min)
- **Optuna optimization** with smart layer selection (not just persistence)
- **Dual output**: clause types for recommendations + domains for navigation

**Cons**
- **26.2% outlier rate** — higher than HDBSCAN clause (18.3%)
- EVoC v0.3.1 is early beta — less community support than HDBSCAN
- No c-TF-IDF keywords (uses LLM labeling instead)
- More types means more LLM calls for field discovery (977s)

---

## 5. Recommendation

### For new deployments: **EVoC Hybrid**

The hybrid pipeline delivers 79 clause types + 10 domains + 944 fields in 20.7 minutes with the highest quality score (0.920). It combines clause-level intelligence with document-level organization.

```bash
python -m core.evoc_pipeline /path/to/pdfs --mode hybrid --optuna-trials 30
```

### For production systems with existing HDBSCAN: **Keep macro-micro**

Until EVoC reaches v1.0, keep HDBSCAN macro-micro for production. Its lower outlier rate (5.2%) and mature ecosystem make it the safer choice.

### For rapid prototyping: **EVoC macro-micro**

At 11.8 minutes, the EVoC M-M pipeline is the fastest option. Good for iteration.

### Summary

| Goal | Best Pipeline | Why |
|---|---|---|
| **Max clause types** | **EVoC Hybrid** | 79 types vs 13 (HDBSCAN) |
| **Max field coverage** | **EVoC Hybrid** | 944 fields, 10,536 extractions |
| **Highest quality** | **EVoC Hybrid** | 0.920 avg quality score |
| **Best speed** | EVoC M-M | 11.8 min |
| **Lowest outlier rate** | HDBSCAN M-M | 5.2% chunk outliers |
| **Production-ready** | HDBSCAN M-M | Most tested |
| **Fastest iteration** | EVoC M-M | 3x faster than HDBSCAN M-M |

---

## 6. Implementation Priority

| Priority | Action | Impact |
|---|---|---|
| P0 | **Reduce outlier rate** — increase Optuna trials (50+), widen param search | Target <20% |
| P1 | **Per-type micro clustering** — chunk-level HDBSCAN within each clause type | Finer extraction |
| P1 | **Field deduplication** — merge identical fields across clause types | Reduce 944 to ~600 |
| P2 | **Incremental mode** — assign new docs to existing types without re-clustering | Production support |
| P2 | **Add c-TF-IDF** — extract keywords per cluster for interpretability | Better labels |

---

## 7. File Reference

| File | Description |
|---|---|
| `core/cluster_pipeline.py` | Macro-Micro pipeline (HDBSCAN) |
| `core/clause_pipeline.py` | Clause-Type pipeline (HDBSCAN) |
| `core/evoc_pipeline.py` | EVoC pipeline (clause, macro-micro, hybrid modes + Optuna) |
| `data/artifacts/hack_s1/run_386b488d/` | Macro-Micro run |
| `data/artifacts/hack_s2/run_c3ddcd5e/` | Clause-Type run |
| `data/artifacts/evoc_hybrid/run_cd74530e/` | **EVoC Hybrid run (recommended)** |
