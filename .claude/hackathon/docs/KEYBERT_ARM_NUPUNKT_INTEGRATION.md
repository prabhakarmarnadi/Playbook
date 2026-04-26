# KeyBERT + ARM + nupunkt Integration

## Implementation and Design Document

**Date:** 2026-04-26
**Status:** Implemented (all 4 phases complete)

---

## 1. Overview

This integration adds three capabilities to the clustering pipeline via a **two-pass architecture**:

| Capability                        | What it does                                                                                                                                      | Pass   |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| **nupunkt**                       | Replaces regex sentence splitting with legal-text-aware structural decomposition (documents -> blocks -> sentences -> chunks)                     | Pass 1 |
| **KeyBERT fingerprints**          | Provides clause-type context for clustering: keyphrase-augmented embeddings, semi-supervised UMAP, and confidence bias in cluster assignment      | Pass 1 |
| **Association Rule Mining (ARM)** | Discovers clause dependencies, field correlations, and term packages via FP-Growth, feeding into field discovery, extraction, and knowledge graph | Pass 2 |

### Two-Pass Pipeline Flow

```
PASS 1 -- Foundation
  1. Parse PDFs                              (unchanged)
  2. Structural decompose via nupunkt         NEW
  3. Score chunks with KeyBERT fingerprints   NEW
  4. Augment chunk text with clause-type prefix
  5. Embed augmented text
  6. Macro cluster with semi-supervised UMAP  MODIFIED
  7. Micro cluster with semi-supervised UMAP  MODIFIED
  8. Topic merge                              (unchanged)
  9. Assign with KeyBERT confidence bias      MODIFIED

PASS 2 -- Enrichment
  10. Build clause presence matrix from assignments
  11. Mine clause association rules (FP-Growth)  NEW
  12. Mine field-value correlations               NEW
  13. Populate relationship layer                  NEW
  14. Field discovery with ARM context            MODIFIED
  15. Extraction with cross-clause context        MODIFIED
  16. Knowledge graph with ARM edges              MODIFIED
```

---

## 2. Design Decisions

### Why two passes?

ARM needs cluster assignments to build the clause presence matrix, but KeyBERT and nupunkt should run before clustering. The two-pass structure resolves this chicken-and-egg: Pass 1 produces better clusters, Pass 2 mines relationships from those clusters.

### Why KeyBERT as soft priors (not hard classification)?

KeyBERT fingerprints are pre-computed from a fixed 32-type taxonomy. The clustering pipeline may discover different or finer-grained clause types. Using KeyBERT as a confidence _modifier_ (default weight 0.15) rather than a hard classifier preserves the pipeline's ability to discover novel clause types while biasing borderline assignments toward known types.

### Why ARM for relationships (not anomaly detection)?

The primary goal is discovering structural relationships in legal documents -- clause dependencies ("Indemnification implies Limitation of Liability"), field correlations ("notice_period_days correlates with cure_period_days"), and term packages (clause bundles that co-occur). These relationships feed into field discovery, extraction, and the knowledge graph.

### Graceful degradation

Every capability degrades independently:

| Missing Component          | Behavior                                               |
| -------------------------- | ------------------------------------------------------ |
| nupunkt not installed      | Falls back to regex sentence splitting                 |
| Fingerprints file absent   | No augmentation, unsupervised UMAP, no confidence bias |
| mlxtend not installed      | ARM mining skipped, RelationshipLayer returns empty    |
| No cluster assignments yet | ARM skipped, consumers use current behavior            |

---

## 3. Architecture

### New Modules

```
core/
  nupunkt_chunker.py        119 lines   Structural decomposition
  keybert_scorer.py          166 lines   Fingerprint classification + augmentation
  relationship_layer.py      222 lines   Unified ARM query interface
  arm/
    __init__.py                1 line
    configs.py                46 lines   ARM threshold defaults
    arm_miner.py             161 lines   Clause-level FP-Growth
    field_arm_miner.py       108 lines   Field-value FP-Growth
prompts/
  field_discovery_arm.txt     32 lines   ARM-augmented discovery prompt
  field_extraction_arm_conditioned.txt  30 lines  ARM-augmented extraction prompt
```

### Modified Modules

```
config.py                    +18 lines  KeyBERT + ARM config vars
core/store.py                +37 lines  Schema: chunks, clusters columns + arm_rules, arm_term_packages tables
core/cuml_wrappers.py        +12 lines  Semi-supervised UMAP (target/target_weight)
core/cluster_assigner.py     +34 lines  KeyBERT confidence bias + load_keybert_priors()
core/chunker.py              +37 lines  structural_chunk() entry point
core/field_discovery.py      +49 lines  _build_arm_context()
core/extractor.py            +35 lines  _build_extraction_arm_context()
core/context_graph.py        +89 lines  add_arm_relationships()
scripts/validate_pipeline.py +146 lines 5 new test functions
```

### Data Flow Diagram

```
                    +-----------------+
                    |   PDF Parser    |
                    +--------+--------+
                             |
                    +--------v--------+
                    | nupunkt_chunker |  blocks -> sentences -> chunks
                    +--------+--------+  each chunk carries block_id, sentence_ids
                             |
                    +--------v--------+
                    | keybert_scorer  |  classify -> augment text -> get UMAP labels
                    +--------+--------+
                             |
              +--------------+--------------+
              |                             |
    +---------v----------+     +-----------v-----------+
    | Embed augmented    |     | Semi-supervised UMAP  |
    | text (MiniLM/Nomic)|     | (target=keybert_labels|
    +--------------------+     |  target_weight=0.3)   |
              |                +-----------+-----------+
              |                            |
              +----------+-----------------+
                         |
              +----------v-----------+
              | HDBSCAN clustering   |
              +----------+-----------+
                         |
              +----------v-----------+
              | cluster_assigner     |  adjusted_sim = base + keybert_bias
              +----------+-----------+
                         |
         ================|==================  Pass 2 boundary
                         |
              +----------v-----------+
              | build_clause_        |  binary matrix: docs x clause_types
              | presence_matrix      |
              +----------+-----------+
                         |
              +----------v-----------+
              |   FP-Growth ARM      |  global rules, per-cluster rules,
              |   (mlxtend)          |  term packages, field correlations
              +----------+-----------+
                         |
              +----------v-----------+
              | RelationshipLayer    |  persist to DuckDB, query interface
              +--+------+-------+---+
                 |      |       |
        +--------v+  +--v---+ +-v-----------+
        | field_  |  |extrac| | context_    |
        |discovery|  |tor   | | graph       |
        +---------+  +------+ +-------------+
```

---

## 4. Module Reference

### 4.1 nupunkt_chunker.py

**Purpose:** Three-level structural decomposition of legal text.

**Key function:**

```python
def structural_decompose(text: str) -> list[dict]:
    """Returns:
    [
        {
            "block_id": "blk_...",
            "block_type": "clause_body" | "heading" | "definition" | "recital",
            "section_number": "14.3" | None,
            "text": "full block text",
            "sentences": [
                {"sentence_id": "sent_...", "text": "...", "char_offset": int},
            ]
        },
    ]
    """
```

**Block detection:** Double-newline paragraph splitting, then classified by pattern:

- `ARTICLE/SECTION/Part` prefix -> `heading`
- `"Term" means/shall mean` -> `definition`
- `WHEREAS/RECITAL/PREAMBLE` -> `recital`
- Everything else -> `clause_body`

**Section numbers:** Extracted via `^\d+(?:\.\d+)*\s` regex (handles "14.3", "5.36.1").

**Sentence splitting:** nupunkt `sent_tokenize()` when available (handles legal citations like "42 U.S.C. 1983", section numbers, abbreviations at ~10M chars/sec). Falls back to regex splitting with abbreviation protection.

**Integration:** `chunker.py` exposes `structural_chunk()` which calls `structural_decompose()` first, then applies `sliding_window_chunk()` within each block. Each chunk inherits `block_id`, `sentence_ids[]`, `block_type`, and `section_number`.

### 4.2 keybert_scorer.py

**Purpose:** Classify text against 32 pre-computed clause-type fingerprints and provide three signals for clustering.

**Singleton pattern:** `KeyBERTScorer.get_instance()` (same as `Embedder.get_instance()`).

**Fingerprint source:** `data/models/clause_fingerprints.json` (copied from keyBERTClause). Contains per-clause-type mean embeddings computed from KeyBERT keyphrases. Uses the same MiniLM model the fingerprints were built with.

**Three integration signals:**

| Signal            | Method                                                                                     | Used by                              |
| ----------------- | ------------------------------------------------------------------------------------------ | ------------------------------------ |
| Text augmentation | `augment_text(text)` -> `[Confidentiality \| confidential, disclosure, nda] original text` | Embedder (pre-embedding)             |
| UMAP labels       | `get_label(text)` -> int index or -1 (unlabeled)                                           | `make_umap(target_weight=0.3)`       |
| Confidence bias   | `score_all(text)` -> dict of 32 similarities                                               | `cluster_assigner.assign_document()` |

**Batch API:** `batch_classify()`, `batch_augment()`, `batch_labels()` for efficient processing of chunk lists.

### 4.3 ARM Module (core/arm/)

**arm_miner.py** -- Clause-level FP-Growth:

```python
# Build binary matrix from cluster assignments
matrix = build_clause_presence_matrix({"doc1": ["Termination", "Notice"], ...})

# Mine global rules
rules = mine_global(matrix)
# -> [{"rule_id": "R_abc", "antecedent": ["Termination"], "consequent": ["Notice"],
#      "support": 0.45, "confidence": 0.89, "lift": 2.1, "rule_type": "clause_dependency"}]

# Mine per-cluster with size-adaptive thresholds
cluster_rules = mine_per_cluster(matrix, {"doc1": 0, "doc2": 1, ...})
# -> {0: [rules], 1: [rules]}

# Discover term packages (3+ clause bundles)
packages = discover_term_packages(matrix)
# -> [{"package_id": "TP_xyz", "clause_types": ["Termination", "Notice", "Survival"],
#      "support": 0.35, "size": 3}]
```

**Cluster-size-adaptive thresholds:**

- Large (>=100 docs): `min_support=0.15`
- Medium (>=50 docs): `min_support=0.20`
- Small (>=20 docs): `min_support=0.25`

**field_arm_miner.py** -- Field-value correlations:

```python
# Build itemset matrix from extraction outputs
matrix = build_field_itemsets({"doc1": {"Termination": {"notice_days": 30}}, ...})
# Columns: "Termination.notice_days=31-60d" (binned)

# Mine field rules
rules = mine_field_rules(matrix)
# -> [{"rule_type": "cross_clause_field", ...}]  # when antecedent/consequent span clause types
# -> [{"rule_type": "field_correlation", ...}]    # within same clause type
```

**Value normalization:** Booleans -> "true"/"false", numerics -> binned (0-7d, 8-30d, 31-60d, ..., 365d+), strings -> lowercase truncated to 50 chars, lists -> first 3 items joined with "+".

### 4.4 relationship_layer.py

**Purpose:** Unified query interface that downstream consumers call. Abstracts away ARM internals.

**Query API:**

| Method                                            | Returns                                         | Consumer                       |
| ------------------------------------------------- | ----------------------------------------------- | ------------------------------ |
| `get_related_clause_types(clause_type)`           | `[{clause_type, confidence, lift, direction}]`  | field_discovery, context_graph |
| `get_implied_fields(clause_types)`                | `[{field_item, confidence, lift, source_rule}]` | field_discovery, extractor     |
| `get_field_correlations(field_name, cluster_id?)` | `[{antecedent, consequent, confidence, lift}]`  | extractor                      |
| `get_term_packages()`                             | `[{package_id, clause_types, support, size}]`   | field_discovery, context_graph |
| `get_obligation_chains(clause_type)`              | `[rules with rule_type="obligation_chain"]`     | context_graph                  |

**Mining API:**

| Method                                                | Input                              | Persists to                      |
| ----------------------------------------------------- | ---------------------------------- | -------------------------------- |
| `mine_from_assignments(assignments, labels?, run_id)` | `{doc_id: [clause_types]}`         | `arm_rules`, `arm_term_packages` |
| `mine_field_correlations(extractions, run_id)`        | `{doc_id: {clause: {field: val}}}` | `arm_rules`                      |

**Caching:** Rules and packages are loaded once from DuckDB on first query, cached in memory. Cache is invalidated after mining operations.

### 4.5 Downstream Integration

**field_discovery.py** -- `_build_arm_context()` formats ARM data into prompt-ready strings:

```python
ctx = _build_arm_context("Termination", relationship_layer=layer)
# ctx = {
#   "related_clauses": "- Notice (confidence: 0.91, lift: 2.1, implies)\n...",
#   "field_correlations": "- Termination.notice_days=31-60d (confidence: 0.85)\n...",
#   "term_packages": "- {Termination, Notice, Survival} (support: 0.35)\n...",
# }
```

Injected into `prompts/field_discovery_arm.txt` template. Falls back to "No ... available." when no relationship layer.

**extractor.py** -- `_build_extraction_arm_context()` provides cross-clause field hints:

```python
ctx = _build_extraction_arm_context("Termination", relationship_layer=layer)
# ctx = {
#   "cross_clause_fields": "- Notice.notice_days=31-60d (confidence: 0.85)\n...",
#   "field_correlations": "- Termination.notice -> Notice.cure (lift: 3.2)\n...",
# }
```

**context_graph.py** -- `add_arm_relationships()` creates three edge types:

| Edge Type          | Source           | Target            | Key Properties            |
| ------------------ | ---------------- | ----------------- | ------------------------- |
| `DEPENDS_ON`       | clause_type node | clause_type node  | confidence, lift, support |
| `FIELD_CORRELATES` | field item       | field item        | lift, confidence          |
| `BUNDLED_WITH`     | clause_type node | term_package node | support, package_size     |

Uses the existing `_add_or_increment_fact()` API with an Episode for provenance tracking.

---

## 5. Schema Additions

All additions are idempotent (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` / `CREATE TABLE IF NOT EXISTS`).

### Modified Tables

**chunks:**
| Column | Type | Source |
|---|---|---|
| `block_id` | VARCHAR | nupunkt structural decomposition |
| `sentence_ids` | VARCHAR (JSON) | nupunkt sentence IDs overlapping this chunk |
| `keybert_scores` | VARCHAR (JSON) | 32-dim clause-type similarity vector |
| `augmented_text` | VARCHAR | Text with KeyBERT prefix |

**clusters:**
| Column | Type | Source |
|---|---|---|
| `dominant_clause_type` | VARCHAR | Top KeyBERT clause type for cluster |
| `clause_type_distribution` | VARCHAR (JSON) | Full distribution across 32 types |

### New Tables

**arm_rules:**
| Column | Type | Description |
|---|---|---|
| `rule_id` | VARCHAR PK | Unique rule identifier |
| `antecedent` | VARCHAR (JSON) | Sorted list of antecedent items |
| `consequent` | VARCHAR (JSON) | Sorted list of consequent items |
| `rule_type` | VARCHAR | `clause_dependency`, `field_correlation`, `cross_clause_field`, `obligation_chain` |
| `support` | DOUBLE | Fraction of docs containing both |
| `confidence` | DOUBLE | P(consequent \| antecedent) |
| `lift` | DOUBLE | Observed / expected co-occurrence |
| `cluster_id` | VARCHAR | NULL for global rules |
| `pipeline_run_id` | VARCHAR | Run provenance |

**arm_term_packages:**
| Column | Type | Description |
|---|---|---|
| `package_id` | VARCHAR PK | Unique package identifier |
| `clause_types` | VARCHAR (JSON) | Sorted list of clause types in bundle |
| `support` | DOUBLE | Fraction of docs containing all |
| `size` | INTEGER | Number of clause types |
| `cluster_id` | VARCHAR | NULL for global packages |
| `pipeline_run_id` | VARCHAR | Run provenance |

---

## 6. Configuration

All new variables are in `config.py`, read from environment with defaults:

### KeyBERT

| Variable                           | Default                                | Purpose                                                     |
| ---------------------------------- | -------------------------------------- | ----------------------------------------------------------- |
| `KEYBERT_FINGERPRINTS_PATH`        | `data/models/clause_fingerprints.json` | Pre-computed fingerprints                                   |
| `KEYBERT_CLASSIFICATION_THRESHOLD` | `0.3`                                  | Minimum similarity to assign a clause label                 |
| `KEYBERT_UMAP_TARGET_WEIGHT`       | `0.3`                                  | Semi-supervised UMAP balance (0=unsupervised, 1=supervised) |
| `KEYBERT_PRIOR_WEIGHT`             | `0.15`                                 | Confidence bias magnitude in cluster_assigner               |
| `KEYBERT_AUGMENT_TOP_K`            | `3`                                    | Keyphrases in text augmentation prefix                      |

### ARM

| Variable                   | Default | Purpose                         |
| -------------------------- | ------- | ------------------------------- |
| `ENABLE_ARM_ENRICHMENT`    | `true`  | Toggle Pass 2 enrichment        |
| `ARM_MIN_SUPPORT`          | `0.20`  | Clause-level minimum support    |
| `ARM_MIN_CONFIDENCE`       | `0.7`   | Clause-level minimum confidence |
| `ARM_MIN_LIFT`             | `1.8`   | Clause-level minimum lift       |
| `FIELD_ARM_MIN_SUPPORT`    | `0.05`  | Field-level minimum support     |
| `FIELD_ARM_MIN_CONFIDENCE` | `0.5`   | Field-level minimum confidence  |
| `FIELD_ARM_MIN_LIFT`       | `1.5`   | Field-level minimum lift        |

---

## 7. Dependencies

Added to `pyproject.toml`:

```toml
nupunkt = {version = ">=0.6.0", python = ">=3.11"}
keybert = ">=0.8.0"
mlxtend = ">=0.23.0"
```

**Data file:** `data/models/clause_fingerprints.json` copied from `/mnt/data/code/keyBERTClause/models/`. Contains 32 clause-type fingerprints with mean embeddings and keyphrases.

---

## 8. Validation

Five new test functions in `scripts/validate_pipeline.py`:

| Test                              | What it validates                                                                                                                             |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_nupunkt_chunker()`          | Structural decomposition: block/sentence hierarchy, section number extraction, legal abbreviation handling                                    |
| `test_keybert_scorer()`           | Fingerprint loading, classification (similarity range 0-1), text augmentation (prefix preservation), label generation                         |
| `test_arm_miner()`                | FP-Growth on synthetic 100-doc dataset: matrix construction, global rule mining, term package discovery                                       |
| `test_relationship_layer()`       | End-to-end: mine -> persist to DuckDB -> query related clause types and term packages                                                         |
| `test_relationship_integration()` | Downstream consumer fallbacks: `_build_arm_context()` and `_build_extraction_arm_context()` return correct defaults when no RelationshipLayer |

Run with:

```bash
poetry run python scripts/validate_pipeline.py --skip-llm
```

---

## 9. Artifact Layout

ARM outputs persist alongside existing pipeline artifacts:

```
data/artifacts/<account_id>/<run_id>/
  macro_model/          (existing)
  micro_models/         (existing)
  embeddings.npz        (existing)
  run_manifest.json     (existing -- extended with ARM metadata)
  arm_rules.json        (new -- clause + field rules)
  term_packages.json    (new -- frequent clause bundles)
```

---

## 10. Commit History

```
7e58e9f feat: wire nupunkt, KeyBERT augmentation, and ARM into evoc_pipeline.py
3a4fe44 feat: wire nupunkt, KeyBERT, and ARM into cluster_pipeline.py
7b6fece docs: add implementation and design document for KeyBERT+ARM+nupunkt integration
af6c11b feat: add ARM-augmented prompts, downstream consumer integration, and Phase 4 validation
0a562cb feat: add ARM mining module and RelationshipLayer
f9eacf9 feat: add KeyBERT scorer, semi-supervised UMAP, and confidence bias integration
13fb70d feat: add nupunkt structural decomposition, chunker integration, and validation
1784d66 feat: add nupunkt, keybert, mlxtend dependencies for integration
```

---

## 11. Pipeline Wiring (Completed)

### cluster_pipeline.py (134 lines added)

| Stage           | What was wired                                                                                                                                                                    |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stage 2 (Chunk) | `nupunkt_available()` guard: uses `structural_chunk()` when available, falls back to `sliding_window_chunk()`                                                                     |
| Stage 3 (Embed) | `KeyBERTScorer.batch_augment(chunk_texts)` runs before embedding, replacing raw text with clause-type-prefixed text                                                               |
| Stage 4 (Macro) | `make_umap(target_weight=KEYBERT_UMAP_TARGET_WEIGHT)` enables semi-supervised UMAP; `batch_labels(summaries)` passed as `y=` to `fit_transform`                                   |
| Stage 5 (Micro) | Same semi-supervised UMAP pattern per domain, both optimized and small-domain paths                                                                                               |
| Stage 5.5 (ARM) | New stage after micro-clustering: builds `doc_clause_types` from quality report, calls `RelationshipLayer.mine_from_assignments()`, saves `arm_rules.json` + `term_packages.json` |
| Manifest        | `arm_enrichment` field added                                                                                                                                                      |

### evoc_pipeline.py (105 lines added)

| Stage           | What was wired                                                                                                                                                             |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stage 2 (Chunk) | Same nupunkt guard as cluster_pipeline                                                                                                                                     |
| Stage 3 (Embed) | Same KeyBERT batch_augment pattern                                                                                                                                         |
| ARM stage       | Queries DuckDB directly (`cluster_assignments -> chunks -> agreements -> clusters`) since EVoC's internal tracking differs. Same `mine_from_assignments` + artifact saving |
| Manifest        | `arm_enrichment` field added                                                                                                                                               |

**Note:** EVoC does not use UMAP, so no semi-supervised UMAP wiring for evoc_pipeline.py.

---

## 12. Remaining Work

1. **Evaluate:** Run the eval harness (`python -m eval.runner`) before and after to measure clustering quality impact (silhouette, coherence, DBCV).

2. **Tune:** Adjust `KEYBERT_UMAP_TARGET_WEIGHT` (currently 0.3) and `KEYBERT_PRIOR_WEIGHT` (currently 0.15) based on eval results.

3. **KeyBERT confidence bias wiring:** The `cluster_assigner.py` now has `load_keybert_priors()` — call it from the pipeline orchestrators when using incremental assignment mode.

4. **ARM -> field discovery wiring:** Pass the `RelationshipLayer` instance to `run_field_discovery()` so it can call `_build_arm_context()` with real data instead of the fallback strings.
