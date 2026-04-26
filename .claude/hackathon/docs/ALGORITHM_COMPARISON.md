# Clustering Algorithm Comparison: HDBSCAN vs EVoC

> **Dataset**: 96–98 public legal agreements, ~3,000–3,159 chunks, ~2,002 valid clauses  
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

## 2. Five Pipeline Runs Compared

| Metric | HDBSCAN M-M | HDBSCAN Clause | EVoC M-M | EVoC Clause | **EVoC Hybrid** |
|---|---|---|---|---|---|
| **Pipeline** | `cluster_pipeline.py` | `clause_pipeline.py` | `evoc --mode mm` | `evoc --mode clause` | `evoc --mode hybrid` |
| **Algorithm** | UMAP + HDBSCAN + Optuna | UMAP + HDBSCAN | EVoC | EVoC | **EVoC + Optuna** |
| **Domains** | 7 | 1 | 8 | 1 | **10** |
| **Clause Types** | — | 13 | — | 7 | **79** |
| **Clusters** | 80 | 13 | 22 | 7 | **79** |
| **Total fields** | 918 | 158 | 271 | 78 | **944** |
| **Total extractions** | 10,228 | 11,837 | 5,483 | 11,355 | **10,536** |
| **Avg quality** | 0.908 | 0.883 | 0.906 | 0.883 | **0.920** |
| **Outlier rate** | 5.2% (chunks) | 18.3% (clauses) | 33 docs | 42.8% (clauses) | **26.2% (clauses)** |
| **Total time** | 33.4 min | 22.2 min | 11.8 min | 14.4 min | **20.7 min** |

### Key Takeaway

The **EVoC Hybrid** pipeline with Optuna optimization is the new recommended approach. It produces:
- **79 clause types** (vs 13 HDBSCAN, 7 EVoC) — 6x more than HDBSCAN clause mode
- **10 document domains** — hierarchical document organization
- **944 fields** — matching HDBSCAN M-M's field coverage (918)
- **0.920 avg quality** — the highest across all runs
- **26.2% outlier rate** — down from 42.8% in the original EVoC clause run

---

## 3. What Changed: EVoC V1 to V2 (Hybrid)

The original EVoC clause run had 3 critical problems. Here is what we fixed:

| Problem | EVoC V1 (old) | EVoC V2 Hybrid (new) | Fix |
|---|---|---|---|
| **Layer selection** | `argmax(persistence)` picked coarse layer | Scoring-based selection (silhouette + cosine + outlier ratio) | `_select_best_layer()` evaluates each layer |
| **base_min_cluster_size** | `n//50 = 40` (too large) | Optuna finds optimal (best: **6**) | Search range [3, 25] via TPE sampler |
| **noise_level** | 0.5 (default) | Optuna finds optimal (**0.30**) | Search range [0.1, 0.6] |
| **Merge threshold** | 0.93 (too aggressive, collapsed 13 to 7) | 0.96 (only merges true duplicates) | Raised from 0.93 to 0.96 |
| **Hyperparam tuning** | None | **30-trial Optuna** optimization | New `evoc_optimize()` with scoring engine |
| **Document domains** | Not available | **10 domains** via macro clustering | New `--mode hybrid` |

### Optuna Best Parameters Found

```
base_min_cluster_size: 6    (was 40 — 85% reduction!)
noise_level: 0.302          (was 0.5 — more aggressive clustering)
n_neighbors: 25             (was 15 — wider neighborhood)
min_samples: 4              (was auto)
```

---

## 4. Speed Comparison

| Stage | HDBSCAN M-M | HDBSCAN Clause | EVoC M-M | EVoC Clause | **EVoC Hybrid** |
|---|---|---|---|---|---|
| Parse + Chunk + Embed | ~92s | ~99s | ~93s | ~92s | ~93s |
| **Clustering** | **973s** | **468s** | **100s** | **75s** | **~160s** |
| **Field Discovery** | **917s** | **718s** | **466s** | **645s** | **977s** |
| **Total** | **2,006s** | **1,334s** | **708s** | **862s** | **1,241s** |

### Speed Analysis

- **EVoC Hybrid clustering** (incl. 30 Optuna trials + macro): ~160s — **6x faster** than HDBSCAN M-M (973s)
- **Total time**: 20.7 min — **38% faster** than HDBSCAN M-M despite producing comparable field coverage
- **Optuna overhead**: 30 EVoC trials complete in ~90s (vs 965s for HDBSCAN Optuna across 7 domains) because EVoC has no UMAP step
- **Field discovery** dominates at 977s (79 clause types x ~12 fields each)

---

## 5. Cluster Quality Comparison

### EVoC Hybrid: 79 Clause Types + 10 Domains

The hybrid pipeline provides **two views** of the same data:

**View 1 — Clause Types** (79 types, cross-document):
Clause types represent semantic patterns that recur across different agreements. Sample types:
- Change Order Price Adjustments (clauses about pricing changes)
- Data Sharing Agreement (data protection and sharing terms)
- Employee IP Assignment (intellectual property clauses)
- Confidentiality Obligations (NDA terms)
- Insurance Provisions (coverage requirements)
- Notice Provisions (how parties communicate)
- Contract Amendment Terms (modification procedures)
- Severability Clauses (partial invalidity handling)

**View 2 — Document Domains** (10 domains):
Documents grouped by overall agreement type. Provides hierarchical navigation.

### Quality Score Comparison

| Metric | HDBSCAN M-M | HDBSCAN Clause | EVoC Clause (old) | **EVoC Hybrid** |
|---|---|---|---|---|
| Avg quality | 0.908 | 0.883 | 0.883 | **0.920** |
| Outlier rate | 5.2% | 18.3% | 42.8% | **26.2%** |
| Cluster count | 80 | 13 | 7 | **79** |

The hybrid's 0.920 quality is the highest across all runs. Optuna found parameters that produce tightly coherent clusters while maintaining good coverage.

---

## 6. Field Discovery Impact

| Approach | Fields | Extractions | Ext/Field | Fields/Cluster |
|---|---|---|---|---|
| HDBSCAN Macro-Micro | 918 | 10,228 | 11.1 | 11.5 |
| HDBSCAN Clause | 158 | 11,837 | 74.9 | 12.2 |
| EVoC Macro-Micro | 271 | 5,483 | 20.2 | 12.3 |
| EVoC Clause (old) | 78 | 11,355 | 145.6 | 11.1 |
| **EVoC Hybrid** | **944** | **10,536** | **11.2** | **11.9** |

### Analysis

- **EVoC Hybrid matches HDBSCAN M-M** on field count (944 vs 918) — the 79 clause types provide enough granularity for comprehensive field discovery
- **Extraction density** (11.2/field) is comparable to HDBSCAN M-M (11.1/field) — fields are neither too broad nor too narrow
- **Universal fields**: 69 fields tagged as universal (appear across multiple types) — "effective_date", "termination_notice_period", "governing_law", etc.

---

## 7. Pros and Cons

### HDBSCAN (UMAP + HDBSCAN via BERTopic)

**Pros**
- Lower outlier rates (5.2% in M-M, 18.3% in clause mode)
- Mature c-TF-IDF keyword extraction per cluster
- BERTopic's outlier reduction strategies (threshold-based reassignment)
- Optuna per-domain tuning for macro-micro mode
- Well-tested in production NLP pipelines

**Cons**
- **Slow**: 33.4 min (macro-micro) due to UMAP + Optuna per domain
- 6 hyperparameters to tune (3 UMAP + 3 HDBSCAN)
- Only 13 clause types too coarse for field extraction
- 80 macro-micro clusters have significant field duplication

### EVoC Hybrid (with Optuna)

**Pros**
- **79 clause types + 10 domains** — best of both architectures
- **Highest quality score** (0.920) across all runs
- **944 fields** — matching HDBSCAN M-M's comprehensive coverage
- **38% faster** than HDBSCAN M-M (20.7 min vs 33.4 min)
- **Optuna tunes only 4 params** (vs 6 for UMAP+HDBSCAN) — faster convergence
- **Smart layer selection** using silhouette + cosine scoring instead of persistence
- **Dual output**: clause types for recommendations + domains for navigation
- No UMAP dimensionality reduction needed

**Cons**
- **26.2% outlier rate** still higher than HDBSCAN's 18.3% (clause mode)
- 79 types may over-segment some areas (though merge threshold handles near-duplicates)
- EVoC v0.3.1 is early beta — API may change
- No c-TF-IDF keyword extraction (uses LLM labeling instead)
- Field discovery takes longer with 79 types vs 13

---

## 8. Recommendation

### Overall winner: **EVoC Hybrid**

The hybrid pipeline with Optuna optimization is the recommended approach for new deployments:

| Goal | Best Pipeline | Why |
|---|---|---|
| **Max field coverage** | **EVoC Hybrid** | 944 fields, 79 clause types |
| **Fastest pipeline** | EVoC M-M | 11.8 min, but only 271 fields |
| **Clause recommendations** | **EVoC Hybrid** | 79 types vs HDBSCAN's 13 |
| **Document organization** | **EVoC Hybrid** | 10 domains + clause types |
| **Highest quality** | **EVoC Hybrid** | 0.920 avg quality |
| **Lowest outlier rate** | HDBSCAN M-M | 5.2% chunk outlier rate |
| **Legacy production** | HDBSCAN M-M | Most tested, proven in production |

### Migration path

1. **New projects**: Use EVoC Hybrid (`python -m core.evoc_pipeline <pdfs> --mode hybrid`)
2. **Existing HDBSCAN pipelines**: Keep until EVoC v1.0 stabilizes
3. **Speed-critical batch jobs**: Use EVoC M-M (11.8 min, 271 fields)

---

## 9. Remaining Improvements

| Priority | Action | Expected Impact |
|---|---|---|
| P0 | **Reduce outlier rate further** — try Optuna with 50+ trials, wider param range | Target <20% outlier rate |
| P1 | **Add per-type micro clustering** — chunk-level HDBSCAN within each clause type | Finer extraction without over-segmentation |
| P1 | **Cross-type field dedup** — merge identical fields across clause types | Reduce 944 to ~600 unique fields |
| P2 | **Add c-TF-IDF to EVoC** — extract keywords per cluster without BERTopic | Improve cluster interpretability |
| P2 | **Incremental clustering** — assign new documents to existing types/domains | Support production incremental ingestion |

---

## 10. File Reference

| File | Description |
|---|---|
| `core/cluster_pipeline.py` | HDBSCAN Macro-Micro pipeline |
| `core/clause_pipeline.py` | HDBSCAN Clause-type pipeline |
| `core/evoc_pipeline.py` | EVoC pipeline (clause, macro-micro, hybrid modes + Optuna) |
| `data/artifacts/hack_s1/run_386b488d/` | HDBSCAN Macro-Micro run |
| `data/artifacts/hack_s2/run_c3ddcd5e/` | HDBSCAN Clause run |
| `data/artifacts/evoc_mm/run_af3d504f/` | EVoC Macro-Micro run |
| `data/artifacts/evoc_clause/run_b22d286b/` | EVoC Clause run (original, pre-tuning) |
| `data/artifacts/evoc_hybrid/run_cd74530e/` | **EVoC Hybrid run (recommended)** |
