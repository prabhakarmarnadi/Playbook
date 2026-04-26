# Pipeline Comparison: Macro-Micro vs Clause-Type Clustering

> **Dataset**: 98 public legal agreements, 3,159 chunks  
> **Hardware**: 2× NVIDIA A100 80GB, Azure OpenAI gpt-4o-mini  
> **Embedding model**: nomic-embed-text-v1.5 (768-dim, CUDA)

---

## 1. Architecture Overview

### Macro-Micro Pipeline (`core/cluster_pipeline.py`)

```
Documents → Macro Cluster (by document embedding) → 7 domains
     └→ Per-domain Micro Cluster (by chunk embedding) → 80 clusters
          └→ Field Discovery (per cluster) → 918 fields, 10,228 extractions
```

- **Stage 1 — Macro**: BERTopic/HDBSCAN on document-level embeddings → 7 coarse domains
- **Stage 2 — Micro**: Per-domain BERTopic/HDBSCAN with Optuna tuning on chunk embeddings → 80 clusters
- **Field discovery**: 15 fields max per cluster, 5 async workers, confidence filtering

### Clause-Type Pipeline (`core/clause_pipeline.py`)

```
Documents → Clause Extraction (structural split) → 2,002 valid clauses
     └→ Clause Embedding (mean of chunk embeddings) → Flat HDBSCAN → 13 clause types
          └→ Field Discovery (per clause type) → 158 fields, 11,837 extractions
```

- **Single stage**: Computes clause-level embeddings (mean over constituent chunks), clusters ALL clauses corpus-wide in a single HDBSCAN pass
- **LLM labeling**: Azure OpenAI labels each clause type with descriptive name
- **Complete-linkage merge**: Near-duplicate types merged at 0.93 cosine threshold
- **Outlier assignment**: Unassigned clauses mapped to nearest type

---

## 2. Head-to-Head Metrics

| Metric | Macro-Micro | Clause-Type | Notes |
|---|---|---|---|
| **Groups** | 7 domains → 80 clusters | 13 clause types | Macro-micro produces far more granular clusters |
| **Total fields** | 918 | 158 | More clusters = more field discovery calls |
| **Unique field names** | 823 | 146 | 95 fields duplicated across clusters in macro-micro |
| **Field duplication** | 56 names in 2+ clusters (max 10×) | 7 names in 2+ clusters | Macro-micro has significant cross-cluster field overlap |
| **Universal fields** | 50 instances (deduped) | 7 instances | |
| **Extractions** | 10,228 | 11,837 | Clause-type produces MORE extractions with FEWER fields |
| **Extractions/field** | 11.1 avg | 74.9 avg | Clause-type fields are far more reusable |
| **Low-confidence filtered** | 0 | 269 | Clause-type has stricter grounding |
| **Outlier handling** | 6 docs, 149 chunks (5.2%) | 366 clauses (18.3%) | Higher outlier rate in clause approach |
| **Avg quality score** | 0.908 across domains | 0.883 across types | Macro-micro shows higher average |
| **Total time** | 33.4 min | 22.2 min | **34% faster** |

### Stage Timing Breakdown

| Stage | Macro-Micro | Clause-Type |
|---|---|---|
| Parse | 0.5s | 0.5s |
| Chunk | 34.1s | 34.2s |
| Embed | 57.6s | 64.4s |
| Persist (DB + Lance) | 19.8s | 46.0s |
| **Clustering** | **973.1s** (macro 8s + micro 965s) | **468.2s** |
| **Field Discovery** | **917.0s** (80 clusters) | **717.6s** (13 clusters) |
| **Total** | **2,006s** | **1,334s** |

Clause-type clustering saves ~11 min by eliminating Optuna per-domain optimization and reducing field discovery calls from 80 to 13.

---

## 3. Cluster Quality Comparison

### Macro-Micro: 7 Domains

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
| Memorandum of Understanding / Project Development | 235 | 51 | 10 | 0.865 |
| Independent Contractor / Contractor Relationship | 186 | 76 | 10 | 0.893 |
| Terms and Conditions of Sale / Order Acceptance | 113 | 10 | 14 | 0.886 |
| Loan and Guarantee Agreement / Recitals | 89 | 4 | 12 | 0.891 |
| Definitions / Settlement Rate Options | 87 | 4 | 14 | 0.868 |
| Table of Contents / Definitions | 84 | 19 | 10 | 0.874 |
| Property Description / Boundary Amendment | 77 | 2 | 12 | 0.867 |
| Lease Compliance / Tenant Certification | 66 | 8 | 12 | 0.873 |
| Special Pricing / Tariff Addendum | 65 | 1 | 14 | 0.991 |
| Consulting Services / Engagement | 61 | 8 | 13 | 0.888 |
| Guaranty Agreement / Securities Recitals | 60 | 1 | 15 | 0.924 |
| Executive Change of Control / Termination | 59 | 6 | 12 | 0.895 |
| Insurance Requirements | 58 | 11 | 10 | 0.877 |

---

## 4. Field Type Distribution

| Field Type | Macro-Micro (918) | Clause-Type (158) |
|---|---|---|
| text | 338 (36.8%) | 62 (39.2%) |
| entity_name | 144 (15.7%) | 22 (13.9%) |
| boolean | 118 (12.9%) | 16 (10.1%) |
| enum | 77 (8.4%) | 15 (9.5%) |
| date | 72 (7.8%) | 14 (8.9%) |
| currency | 63 (6.9%) | 7 (4.4%) |
| integer | 57 (6.2%) | 10 (6.3%) |
| duration | 31 (3.4%) | 9 (5.7%) |
| percentage | 18 (2.0%) | 3 (1.9%) |

Distributions are roughly proportional. Clause-type has slightly higher representation of structured types (date, duration, enum) relative to free-text.

---

## 5. Extraction Coverage

### Macro-Micro
- 80 clusters × ~11.5 fields/cluster = 918 fields
- 10,228 total extractions → ~11.1 per field
- Fields per cluster: min=4, max=15, mean=11.5
- Many fields are cluster-specific and rarely reusable

### Clause-Type
| Clause Type | Fields | Extractions | Ext/Field |
|---|---|---|---|
| Independent Contractor | 10 | 3,857 | 385.7 |
| Memorandum of Understanding | 10 | 3,223 | 322.3 |
| Table of Contents / Definitions | 10 | 1,133 | 113.3 |
| Special Pricing / Tariff | 14 | 887 | 63.4 |
| Definitions / Settlement Rate | 14 | 659 | 47.1 |
| Property Description | 12 | 432 | 36.0 |
| Executive Change of Control | 12 | 324 | 27.0 |
| Lease Compliance | 11 | 273 | 24.8 |
| Guaranty Agreement | 14 | 244 | 17.4 |
| Terms and Conditions | 14 | 241 | 17.2 |
| Loan and Guarantee | 12 | 218 | 18.2 |
| Consulting Services | 12 | 174 | 14.5 |
| Insurance Requirements | 10 | 172 | 17.2 |

Clause-type fields extract far more values per field because they represent cross-document clause patterns rather than narrow topic clusters.

---

## 6. Pros and Cons

### Macro-Micro Pipeline

**Pros**
- Higher average quality score (0.908 vs 0.883) — Optuna tuning optimizes per-domain clustering
- 80 fine-grained clusters capture narrow topic distinctions (e.g., "Limitation of Liability" vs "Termination and Transition Assistance" vs "Notices" vs "Payment Terms")
- More total fields (918) — covers long-tail of rare clause patterns
- Lower outlier rate (5.2% of chunks) — fewer unassigned items
- Domain hierarchy provides natural navigation (domain → cluster → field)

**Cons**
- **7 coarse macro domains** — documents clustered at the wrong granularity; a "Service Contractor Agreement" domain lumps SOWs, payment terms, IP clauses, and termination clauses together
- **No clause awareness** — clusters chunks, not clauses; misses the contractual unit that lawyers think in
- **Significant field duplication** — `effective_date` appears in 10 clusters, `project_name` in 7; 56 field names duplicated across clusters
- **Slow** — 33.4 min due to Optuna optimization per domain (965s for micro clustering alone)
- **Many low-reusability fields** — 11.1 extractions/field avg means most fields fire rarely
- **No clause structure in output** — cannot answer "what clauses does this contract type typically have?"

### Clause-Type Pipeline

**Pros**
- **Clause-level semantics** — groups actual contractual clauses (with titles, numbers, hierarchy) cross-document
- **Highly reusable fields** — 74.9 extractions/field avg; each field is a genuinely common data point
- **More total extractions** (11,837 vs 10,228) with fewer fields — better signal-to-noise
- **34% faster** (22.2 min vs 33.4 min) — single-pass HDBSCAN, no Optuna
- **Minimal field duplication** — only 7 names shared across types
- **Hierarchical output** — clause_types.json provides type → clause → field structure
- **Confidence filtering** — 269 low-confidence extractions removed (source-text grounding)

**Cons**
- **13 types may be too coarse** — "MOU / Project Development Agreement" (235 clauses, 51 agreements) is a catch-all; misses distinctions like "Indemnification" vs "Governing Law" vs "Force Majeure"
- **Higher outlier rate** (18.3%) — 762 of 2,002 clauses are outliers assigned to nearest type, not in export
- **Some types are single-agreement** — "Special Pricing" (1 agreement), "Guaranty" (1 agreement) overfit to individual document structure
- **Lower quality scores** (0.883 avg) — no per-domain Optuna optimization to tune HDBSCAN
- **Fewer total fields** (158 vs 918) — may miss rare but important clauses
- **min_cluster_size=40** (n//50) may be too aggressive — standard legal clauses with <40 occurrences are absorbed as outliers

---

## 7. Recommendation

### For production quality extraction: **Macro-Micro**

The macro-micro approach produces richer field coverage (918 fields) with fine-grained clusters that distinguish narrow legal topics. While it has more field duplication and a coarser top level, the per-domain Optuna optimization yields tighter clusters (0.908 avg quality). For a CLM system that needs to extract every possible data point from diverse contract types, the breadth of 80 clusters is valuable even with some redundancy.

### For clause-level intelligence: **Clause-Type**

If the goal is to understand "what clauses exist across our portfolio" and recommend standard clauses, the clause-type pipeline is superior. Its 13 types map directly to contractual concepts, produce highly reusable fields (75× extraction rate), and deliver this in 34% less time. The hierarchical output (type → clause → field) directly supports clause recommendation workflows.

### Best of both: **Hybrid approach** (recommended)

The ideal pipeline would combine both:

1. **Clause extraction** — Use the structural clause split to identify clause boundaries (already in both pipelines)
2. **Clause-type clustering** — Cluster clauses into types (the clause pipeline's approach) for clause-level recommendations
3. **Per-type micro clustering** — Within each clause type, run fine-grained HDBSCAN on chunk embeddings to capture sub-topics (from the macro-micro approach, but scoped per clause type instead of per domain)
4. **Field discovery** — Run on both clause types (for universal clause fields) and micro clusters (for detailed extraction)

This would give:
- The clause-type hierarchy for recommendations and navigation
- The micro-cluster granularity for comprehensive field extraction
- No document-level macro clustering needed (eliminating the weakest part of the old pipeline)

### Implementation priority

| Priority | Action | Impact |
|---|---|---|
| P0 | **Reduce clause-type outlier rate** — lower `min_cluster_size` from 40 to 15-20 | Captures more clause types, reduces 18.3% outlier rate |
| P0 | **Split the MOU catch-all** — the 235-clause MOU type needs finer subdivision | Better type precision |
| P1 | **Add per-type micro clustering** — run chunk-level HDBSCAN within each clause type | Gets back the fine-grained extraction without macro clustering |
| P2 | **Cross-reference fields** — deduplicate fields that appear across both levels | Eliminates redundancy from hybrid approach |

---

## 8. File Reference

| File | Description |
|---|---|
| `core/cluster_pipeline.py` | Macro-Micro pipeline (restored) |
| `core/clause_pipeline.py` | Clause-Type pipeline |
| `data/artifacts/hack_s1/run_386b488d/` | Macro-Micro run artifacts |
| `data/artifacts/hack_s2/run_c3ddcd5e/` | Clause-Type run artifacts |
