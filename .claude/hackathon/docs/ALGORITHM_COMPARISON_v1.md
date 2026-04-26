# Clustering Algorithm Comparison: HDBSCAN vs EVoC

> **Dataset**: 98 public legal agreements, 3,159 chunks, 2,002 valid clauses  
> **Hardware**: 2× NVIDIA A100 80GB, Azure OpenAI gpt-4o-mini  
> **Embedding model**: nomic-embed-text-v1.5 (768-dim, CUDA)  
> **EVoC version**: 0.3.1, **BERTopic version**: 0.17.4

---

## 1. What is EVoC?

**EVoC** (Embedding Vector Oriented Clustering) is a clustering library specifically designed for high-dimensional embedding vectors. It replaces the conventional UMAP + HDBSCAN two-step approach used in BERTopic with a single-pass algorithm that:

- Works directly on cosine distance (no dimensionality reduction needed)
- Produces **multi-granularity cluster layers** automatically
- Has fewer hyperparameters than UMAP + HDBSCAN combined
- Uses node embedding of nearest-neighbor graphs (related to UMAP) and density-based clustering (related to HDBSCAN)

**HDBSCAN** (via BERTopic) is the established approach: UMAP reduces embeddings to low dimensions, then HDBSCAN finds density-based clusters, followed by c-TF-IDF topic representation.

---

## 2. Four Pipeline Runs Compared

| Metric | HDBSCAN Macro-Micro | HDBSCAN Clause | EVoC Macro-Micro | EVoC Clause |
|---|---|---|---|---|
| **Pipeline** | `cluster_pipeline.py` | `clause_pipeline.py` | `evoc_pipeline.py --mode macro-micro` | `evoc_pipeline.py --mode clause` |
| **Clustering algorithm** | UMAP + HDBSCAN + Optuna | UMAP + HDBSCAN | EVoC | EVoC |
| **Domains** | 7 | 1 | 8 | 1 |
| **Clusters / Types** | 80 | 13 | 22 | 7 |
| **Total fields** | 918 | 158 | 271 | 78 |
| **Total extractions** | 10,228 | 11,837 | 5,483 | 11,355 |
| **Skipped (low conf)** | 0 | 269 | 100 | 732 |
| **Avg quality** | 0.908 | 0.883 | 0.906 | 0.883 |
| **Outlier rate** | 5.2% (chunks) | 18.3% (clauses) | 33 docs outlier | 42.8% (clauses) |
| **Total time** | 33.4 min | 22.2 min | 11.8 min | 14.4 min |

---

## 3. Speed Comparison

| Stage | HDBSCAN M-M | HDBSCAN Clause | EVoC M-M | EVoC Clause |
|---|---|---|---|---|
| Parse | 0.5s | 0.5s | 0.5s | 0.5s |
| Chunk | 34.1s | 34.2s | 34.5s | 34.2s |
| Embed | 57.6s | 64.4s | 58.1s | 57.6s |
| Persist | 19.7s | 46.0s | 45.4s | 46.8s |
| **Clustering** | **973.1s** | **468.2s** | **99.7s** | **75.4s** |
| **Field Discovery** | **917.0s** | **717.6s** | **466.1s** | **644.7s** |
| **Total** | **2,006s** | **1,334s** | **708s** | **862s** |

### Key Speed Findings

- **EVoC clustering is 10× faster than HDBSCAN macro-micro** (99.7s vs 973.1s) — eliminates Optuna tuning entirely
- **EVoC clustering is 6× faster than HDBSCAN clause** (75.4s vs 468.2s) — no UMAP reduction step
- **EVoC macro-micro is the fastest overall** at 11.8 min — 65% faster than HDBSCAN macro-micro
- Field discovery time scales with # clusters: fewer clusters = fewer LLM calls = faster

---

## 4. Cluster Quality Comparison

### Macro-Micro Mode

| Metric | HDBSCAN | EVoC |
|---|---|---|
| Domains | 7 | 8 |
| Clusters | 80 | 22 |
| Avg quality | 0.908 | 0.906 |
| Outlier docs | 6 (6.1%) | 33 (33.7%) |

**HDBSCAN domains** (7): Service Contractor, Data Sharing, Real Estate, Government Procurement, Director Equity, Loan/Guarantee, Software/Hardware

**EVoC domains** (8): Public Program Compliance, Public Works Contracts, Financial/Sovereign, Securities/Governance, Lease/Housing, Municipal Land-Use, Data Sharing, Commercial

EVoC produces **better-labeled domains** (more descriptive names) and one additional domain, but has a **much higher document outlier rate** (33 vs 6 docs). HDBSCAN with Optuna produces 80 fine-grained clusters vs EVoC's 22.

### Clause Mode

| Metric | HDBSCAN | EVoC |
|---|---|---|
| Clause types | 13 | 7 |
| Avg quality | 0.883 | 0.883 |
| Outlier clauses | 366 (18.3%) | 857 (42.8%) |

**HDBSCAN clause types** (13): MOU/Project Development, Independent Contractor, Terms and Conditions, Loan/Guarantee, Definitions, Table of Contents, Property Description, Lease Compliance, Special Pricing, Consulting Services, Guaranty Agreement, Executive Change of Control, Insurance Requirements

**EVoC clause types** (7): Government Resolutions, Employment Agreement Terms, Purchase Terms and Conditions, Table of Contents, Loan Guarantee Agreements, FX Definitions Annex, Special Pricing Arrangements

HDBSCAN finds **nearly 2× more clause types** and has **less than half the outlier rate**. EVoC's clause types are coarser — "Government Resolutions" (458 clauses) and "Employment Agreement Terms" (405 clauses) are very broad catch-alls.

---

## 5. EVoC Macro-Micro: Domain Detail

| Domain | Clusters | Chunks | Quality | Outlier Chunks |
|---|---|---|---|---|
| Commercial Agreements | 3 | 537 | 0.898 | 84 |
| Public Works Service Contracts | 4 | 352 | 0.901 | 41 |
| Lease and Housing Agreements | 3 | 318 | 0.896 | 13 |
| Financial and Sovereign Agreements | 4 | 299 | 0.910 | 42 |
| Data Sharing Agreements | 3 | 211 | 0.908 | 32 |
| Securities and Governance | 1 | 185 | 0.925 | 43 |
| Public Program Compliance | 2 | 95 | 0.896 | 2 |
| Municipal Land-Use Approvals | 2 | 37 | 0.911 | 0 |

Notable clusters: "Contract terms and compliance" (191 chunks), "Terms and Sale Conditions" (184), "Employee IP Assignment" (184), "Lease Compliance Forms" (227), "Ginnie Mae issuer duties" (142).

---

## 6. Field Discovery Impact

| Approach | Fields | Extractions | Ext/Field | Fields/Cluster |
|---|---|---|---|---|
| HDBSCAN Macro-Micro | 918 | 10,228 | 11.1 | 11.5 |
| HDBSCAN Clause | 158 | 11,837 | 74.9 | 12.2 |
| EVoC Macro-Micro | 271 | 5,483 | 20.2 | 12.3 |
| EVoC Clause | 78 | 11,355 | 145.6 | 11.1 |

### Analysis

- **EVoC clause has the highest extraction density** — 145.6 extractions per field. 7 broad clause types means each field captures a wide cross-section of the corpus.
- **HDBSCAN macro-micro has the most fields** (918) but lowest density (11.1/field) — many fields are cluster-specific and rarely reusable.
- **EVoC macro-micro** is a middle ground: 271 fields with 20.2 extractions/field.
- **Confidence filtering**: EVoC clause skipped the most low-confidence extractions (732), suggesting its broader clusters include more heterogeneous content.

---

## 7. Pros and Cons

### HDBSCAN (UMAP + HDBSCAN via BERTopic)

**Pros**
- Higher cluster count → better field coverage for long-tail patterns
- Lower outlier rates in both modes (5.2% and 18.3%)
- Optuna tuning optimizes per-domain HDBSCAN params for best quality
- BERTopic provides c-TF-IDF topic representations (keywords)
- Mature, well-tested in production NLP pipelines

**Cons**
- **Slow**: 33.4 min (macro-micro) due to UMAP + Optuna per domain
- UMAP dimreduction adds complexity and hyperparameters (n_neighbors, n_components, min_dist)
- 80 clusters may be over-segmented — many are hard to distinguish
- Sequential per-domain optimization is the bottleneck

### EVoC (Embedding Vector Oriented Clustering)

**Pros**
- **Dramatically faster**: 10× faster clustering, 65% faster end-to-end
- Works directly on embeddings — no UMAP reduction = no dimension-related tuning
- **Multi-granularity layers built in** — automatically provides coarse + fine views
- Fewer hyperparameters (just `noise_level` and `base_min_cluster_size`)
- Better domain labels (more descriptive domain names via same-quality LLM labeling)
- Near-identical quality scores to HDBSCAN (0.906 vs 0.908)

**Cons**
- **Much higher outlier rates** (42.8% in clause mode, 33.7% doc outlier rate in M-M)
- **Fewer clusters found** (22 vs 80 in M-M, 7 vs 13 in clause) — may miss fine-grained patterns
- **Coarser groupings** — "Government Resolutions" (458 clauses) is too broad
- Fewer total fields discovered (78 vs 158 clause, 271 vs 918 M-M)
- Early beta library (v0.3.1) — API may change, less community support
- No c-TF-IDF keyword extraction (requires separate implementation)
- Higher confidence filtering (732 skipped) suggests noisier cluster boundaries

---

## 8. Recommendation

### For comprehensive field extraction: **HDBSCAN Macro-Micro**

918 fields across 80 fine-grained clusters provides the most complete data extraction coverage. The Optuna optimization produces tightly coherent clusters. Best for production CLM systems that need to capture every possible data point.

### For speed with good-enough quality: **EVoC Macro-Micro** ⭐

22 clusters with 271 fields in just 11.8 min — 65% faster than HDBSCAN. Quality scores are virtually identical (0.906 vs 0.908). The 8 domains have better descriptive labels. Ideal for rapid iteration, prototyping, and large-scale batch processing.

### For clause-level intelligence: **HDBSCAN Clause**

13 clause types vs EVoC's 7, with half the outlier rate (18.3% vs 42.8%). More clause types = finer-grained clause recommendations. The hierarchical clause_types.json output enables "what clauses should this contract have?" workflows.

### Not recommended: **EVoC Clause** (as-is)

7 clause types with 42.8% outlier rate means nearly half the clauses are forced-assigned to their nearest type. The broad types ("Government Resolutions" = 458 clauses) aren't useful for clause recommendations. Reducing `noise_level` from 0.5 to 0.2 might help, but HDBSCAN's BERTopic outlier reduction is more effective here.

### Summary Matrix

| Goal | Best Algorithm | Best Mode | Why |
|---|---|---|---|
| **Max field coverage** | HDBSCAN | Macro-Micro | 918 fields, 80 clusters |
| **Fastest pipeline** | EVoC | Macro-Micro | 11.8 min, near-identical quality |
| **Clause recommendations** | HDBSCAN | Clause | 13 types, 18% outliers |
| **Field reusability** | EVoC | Clause | 145.6 extractions/field |
| **Production quality** | HDBSCAN | Macro-Micro | Most tested, lowest outlier rate |
| **Rapid prototyping** | EVoC | Macro-Micro | 3× faster, good cluster labels |

---

## 9. Potential Improvements

| Priority | Action | Expected Impact |
|---|---|---|
| P0 | **Tune EVoC `noise_level`** — try 0.2-0.3 for clause mode | Reduce 42.8% outlier rate, more clause types |
| P0 | **Fix EVoC M-M doc outliers** — 33 of 98 docs orphaned; adjust macro `base_min_cluster_size` | Better document coverage |
| P1 | **Hybrid approach**: EVoC for speed, then HDBSCAN refinement on ambiguous clusters | Best of both worlds |
| P1 | **Use EVoC multi-layer output** — provide fine+coarse clause types to users | Leverage EVoC's unique multi-granularity |
| P2 | **Add c-TF-IDF to EVoC pipeline** — extract keywords per cluster without BERTopic | Improve cluster interpretability |
| P2 | **EVoC `base_n_clusters`** — set target cluster count for clause mode | Control granularity |

---

## 10. File Reference

| File | Description |
|---|---|
| `core/cluster_pipeline.py` | HDBSCAN Macro-Micro pipeline |
| `core/clause_pipeline.py` | HDBSCAN Clause-type pipeline |
| `core/evoc_pipeline.py` | EVoC pipeline (both modes via `--mode`) |
| `data/artifacts/hack_s1/run_386b488d/` | HDBSCAN Macro-Micro run |
| `data/artifacts/hack_s2/run_c3ddcd5e/` | HDBSCAN Clause run |
| `data/artifacts/evoc_mm/run_af3d504f/` | EVoC Macro-Micro run |
| `data/artifacts/evoc_clause/run_b22d286b/` | EVoC Clause run |
