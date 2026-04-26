# KeyBERT + ARM + nupunkt Integration Design

**Date:** 2026-04-26
**Status:** Approved
**Approach:** Two-Pass Architecture (Foundation + Enrichment)

## Overview

Integrate three capabilities into the clustering pipeline:

1. **nupunkt** — structural decomposition of legal text (blocks → sentences → chunks)
2. **KeyBERT fingerprints** — clause-type-aware clustering context (text augmentation, semi-supervised UMAP, confidence bias)
3. **Association Rule Mining (ARM)** — relationship discovery (clause dependencies, field correlations, obligation chains) feeding into field discovery, knowledge graph, and extraction

The integration follows a two-pass architecture: Pass 1 (Foundation) handles structural decomposition and KeyBERT-enhanced clustering. Pass 2 (Enrichment) mines relationships from cluster assignments and feeds them to downstream consumers.

## Source Projects

- **keyBERTClause** (`/mnt/data/code/keyBERTClause`) — KeyBERT fingerprinting (32 clause types), FP-Growth ARM (clause + field level), deviation scoring
- **nupunkt** — legal-text-aware sentence/paragraph tokenizer (~10M chars/sec, handles section numbers, citations, abbreviations)

---

## Section 1: nupunkt Structural Decomposition

### What Changes

Replace the regex-based `_split_sentences()` in `chunker.py` with a three-level structural decomposition using nupunkt.

### New Module: `core/nupunkt_chunker.py`

### Hierarchy Produced

```
Document
 └── Block (paragraph/section — nupunkt PunktParagraphTokenizer)
      ├── block_id, block_type (clause_body, heading, definition, recital)
      ├── parent_section_number (e.g., "14.3")
      └── Sentence (nupunkt sent_tokenize)
           ├── sentence_id, char_offset, confidence
           └── Chunk (sliding window or semantic, as today)
                ├── chunk_id, token_count
                └── inherits block_id + sentence_ids for traceability
```

### Key Decisions

- nupunkt replaces regex splitting but the existing chunking strategies (`sliding_window_chunk`, `semantic_chunk`) remain — they operate on better-segmented input
- Each chunk carries `block_id` and `sentence_ids[]` metadata, preserving structural context for downstream ARM and extraction
- `chunker.py` gets a new entry point `structural_chunk(text)` that calls nupunkt first, then applies the chosen chunking strategy within each block
- Fallback: if nupunkt is not installed, degrade to current regex splitting (same pattern as `cuml_available()`)

### Schema Impact

`chunks` table in DuckDB gains `block_id TEXT` and `sentence_ids TEXT` (JSON array) columns via idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.

---

## Section 2: KeyBERT as Clustering Context

### What Changes

KeyBERT participates in three stages — pre-embedding context augmentation, semi-supervised UMAP, and post-clustering confidence bias.

### New Module: `core/keybert_scorer.py`

### Fingerprint Source

Reuse the pre-computed fingerprints from keyBERTClause (`models/clause_fingerprints.json` — 32 clause types x 20 keyphrases each, with mean embeddings). No need to retrain.

### Stage A — Keyphrase-Augmented Embedding

```
Raw chunk text: "Party shall not disclose any proprietary information..."
        ↓
keybert_scorer.classify(text) → top clause type + keyphrases
        ↓
Augmented text: "[Confidentiality | confidential, disclosure, non-disclosure,
                  proprietary] Party shall not disclose any proprietary information..."
        ↓
embedder.embed(augmented_text) → 384/768-dim vector
```

The prefix tag gives the embedding model clause-type context, pulling semantically similar clauses closer in vector space. The format `[ClauseType | kw1, kw2, kw3]` is compact enough to not dominate the embedding but distinctive enough to influence it.

### Stage B — Semi-Supervised UMAP

```
chunk_embeddings (n, 384)  +  keybert_labels (n,)
        ↓
UMAP(target=keybert_labels, target_weight=0.3)
        ↓
reduced embeddings that preserve both semantic similarity
AND clause-type grouping
```

- `target_weight` controls the balance: 0.0 = fully unsupervised (current behavior), 1.0 = fully supervised. Default `0.3` — enough to guide structure without forcing it.
- `KEYBERT_UMAP_TARGET_WEIGHT` configurable in `config.py`
- KeyBERT labels are soft: only chunks above a classification threshold (default `0.3`) get a label. Below-threshold chunks get `target=-1` (UMAP treats -1 as unlabeled), so uncertain chunks cluster naturally.
- Applies to both macro and micro clustering UMAP steps via `cuml_wrappers.make_umap()` gaining an optional `target` parameter.

### Stage C — Confidence Bias

The existing three-tier model uses raw cosine similarity to cluster centroids (HIGH >= 0.85, TENTATIVE >= 0.60, NOVEL < 0.60). KeyBERT priors adjust this as a confidence modifier, not a replacement:

```python
base_confidence = cosine_sim(chunk_embedding, centroid)
alignment = keybert_scores[cluster.dominant_clause_type]
prior_adjustment = (alignment - 0.5) * KEYBERT_PRIOR_WEIGHT  # default 0.15
adjusted_confidence = base_confidence + prior_adjustment
```

### Full Flow Through Pipeline

```
1. Parse PDFs (pdf_parser)
2. Structural decompose (nupunkt_chunker) → blocks → sentences → chunks
3. Score chunks (keybert_scorer) → clause types + keyphrases per chunk
4. Augment chunk text with KeyBERT prefix
5. Embed augmented text (embedder)
6. Macro cluster with semi-supervised UMAP (target=doc-level clause types)
7. Micro cluster per domain with semi-supervised UMAP (target=chunk-level clause types)
8. Assign with KeyBERT confidence bias
```

### Key Decisions

- KeyBERT scoring runs _before_ embedding — this is the critical ordering change
- Augmentation prefix format is deterministic and reproducible (alphabetically sorted keyphrases, top `KEYBERT_AUGMENT_TOP_K` keyphrases by score)
- `cuml_wrappers.make_umap()` gains `target` and `target_weight` optional parameters, passing through to both cuML and sklearn UMAP (both support semi-supervised)
- Fingerprints file (`clause_fingerprints.json`) is copied from keyBERTClause into `data/models/` at setup time
- If fingerprints are missing, all three stages degrade gracefully: no augmentation, unsupervised UMAP, no confidence bias
- KeyBERT scoring uses the same MiniLM model the fingerprints were built with, regardless of the pipeline's embedding model
- Clusters gain a `dominant_clause_type` field — derived during micro-clustering by running the cluster's representative chunks through KeyBERT and taking the top clause type
- Scorer is lazy-loaded and cached (same pattern as `Embedder.get_instance()`)

### Schema Impact

- `clusters` table gains `dominant_clause_type TEXT` and `clause_type_distribution TEXT` (JSON) columns
- `chunks` table gains `keybert_scores TEXT` (JSON, 32-dim vector) and `augmented_text TEXT`

### New Config Vars

| Variable                           | Default                                | Purpose                                                     |
| ---------------------------------- | -------------------------------------- | ----------------------------------------------------------- |
| `KEYBERT_FINGERPRINTS_PATH`        | `data/models/clause_fingerprints.json` | Path to fingerprints JSON                                   |
| `KEYBERT_CLASSIFICATION_THRESHOLD` | `0.3`                                  | Minimum similarity to assign a label                        |
| `KEYBERT_UMAP_TARGET_WEIGHT`       | `0.3`                                  | Semi-supervised UMAP balance (0=unsupervised, 1=supervised) |
| `KEYBERT_PRIOR_WEIGHT`             | `0.15`                                 | Confidence bias strength in cluster_assigner                |
| `KEYBERT_AUGMENT_TOP_K`            | `3`                                    | Number of keyphrases in text augmentation prefix            |

---

## Section 3: ARM Relationship Layer

### What Changes

Add a new `core/arm/` module that mines clause and field relationships from pipeline outputs, and a `core/relationship_layer.py` that provides a unified interface for downstream consumers.

### New Modules

```
core/arm/
  __init__.py
  arm_miner.py          — clause-level FP-Growth (ported from keyBERTClause)
  field_arm_miner.py    — field-value correlation mining (ported from keyBERTClause)
  configs.py            — threshold defaults (inlined, not separate JSON files)
core/relationship_layer.py  — unified query interface over ARM outputs
```

### What ARM Discovers

ARM is for relationship discovery, not anomaly detection.

| Relationship Type       | Example                                                                       | Consumer                       |
| ----------------------- | ----------------------------------------------------------------------------- | ------------------------------ |
| Clause dependency       | Indemnification → Limitation of Liability (confidence 0.89)                   | field_discovery, context_graph |
| Term package            | {Termination, Notice, Survival} co-occur as a bundle (support 0.45)           | field_discovery, context_graph |
| Field correlation       | notice_period_days=30 → cure_period_days=30 (lift 3.2)                        | extractor, context_graph       |
| Cross-clause field link | Non-Compete.duration_months correlates with Termination.survival_months       | extractor, field_discovery     |
| Obligation chain        | Confidentiality + IP Rights → Return/Destruction obligation (confidence 0.78) | context_graph                  |

### ARM Inputs

The clause presence matrix is built from cluster assignments after Pass 1 completes. Each document's assigned clause types (from micro-clustering + KeyBERT dominant types) form a binary row. Field correlations use extraction outputs if available, or run incrementally as extractions are produced.

### Relationship Layer Interface

```python
class RelationshipLayer:
    def __init__(self, store: ClusteringStore)

    # Query methods — consumers call these
    def get_related_clause_types(clause_type: str) -> list[dict]
        # Returns co-occurring clause types with confidence/lift

    def get_implied_fields(clause_types: list[str]) -> list[dict]
        # Given present clause types, what fields should we look for?

    def get_field_correlations(field_name: str, cluster_id: str) -> list[dict]
        # What other field values correlate with this field?

    def get_term_packages() -> list[dict]
        # Frequent clause bundles (3+ types)

    def get_obligation_chains(clause_type: str) -> list[dict]
        # What obligations follow from this clause type?

    # Mining methods — called by pipeline orchestrator
    def mine_from_assignments(cluster_assignments: dict) -> None
    def mine_field_correlations(extractions: dict) -> None
```

### Storage

ARM outputs persist in DuckDB via two new tables:

- `arm_rules` — antecedent (JSON), consequent (JSON), rule_type (clause/field/obligation), support, confidence, lift, cluster_id (nullable for global rules)
- `arm_term_packages` — clause_types (JSON), support, size, cluster_id (nullable)

### Key Decisions

- Port the FP-Growth logic from keyBERTClause's `arm_miner.py` and `field_arm_miner.py`, adapting to use `ClusteringStore` instead of CSV files
- Threshold defaults match keyBERTClause: `min_support=0.20, min_confidence=0.7, min_lift=1.8` for clause-level; `min_support=0.05, min_confidence=0.5, min_lift=1.5` for field-level
- Cluster-size-adaptive thresholds preserved (large >= 100, medium >= 50, small >= 20)
- `mlxtend` added as a dependency in `pyproject.toml` for FP-Growth
- RelationshipLayer caches results in memory after first query per session

---

## Section 4: Downstream Consumers

### 4a — Field Discovery (`field_discovery.py`)

Currently, field discovery asks the LLM to discover fields from sample chunks of a single cluster in isolation. With ARM, before calling the LLM, `field_discovery.py` queries `RelationshipLayer.get_related_clause_types()` and `get_implied_fields()` to build relationship context injected into the prompt.

**Before:**

```
"Here are chunks from a Termination clause cluster. What fields exist?"
```

**After:**

```
"Here are chunks from a Termination clause cluster.
 Related clauses: Notice (confidence 0.91), Survival (0.78).
 Known field correlations: notice_period_days <-> cure_period_days.
 Term package: {Termination, Notice, Survival} appears in 45% of contracts.
 Suggest fields for this clause type AND cross-referenced fields
 from related clause types."
```

**New prompt template:** `prompts/field_discovery_arm.txt`

### 4b — Knowledge Graph (`context_graph.py`)

ARM relationships become first-class edges:

| Edge Type            | Source           | Target             | Properties                  |
| -------------------- | ---------------- | ------------------ | --------------------------- |
| `DEPENDS_ON`         | Clause cluster   | Clause cluster     | confidence, lift, support   |
| `BUNDLED_WITH`       | Clause cluster   | Term package node  | support, package_size       |
| `FIELD_CORRELATES`   | Field definition | Field definition   | lift, direction, cluster_id |
| `IMPLIES_OBLIGATION` | Clause cluster   | Obligation cluster | confidence, chain_length    |

**Integration point:** `context_graph.py` gains a new method `add_arm_relationships(relationship_layer)` that queries all relationship types and creates edges. Additive — existing graph-building logic is untouched. Edges are idempotent (upserted by rule_id).

### 4c — Extraction (`extractor.py`)

Currently, extraction is conditioned on a single cluster's field schema. With ARM, it gains cross-clause context:

**Before:**

```python
extract_conditioned(text, cluster_label="Termination",
                    fields=["notice_period", "cure_period"])
```

**After:**

```python
extract_conditioned(text, cluster_label="Termination",
                    fields=["notice_period", "cure_period"],
                    cross_clause_fields=["survival_months"],
                    field_correlations={"notice_period_days":
                      {"correlated_with": "cure_period_days", "lift": 3.2}})
```

**New prompt template:** `prompts/field_extraction_arm_conditioned.txt` — includes a section for cross-clause fields and correlation context. Existing `field_extraction_conditioned.txt` remains as fallback.

### Key Decisions

- All three integrations are additive — if RelationshipLayer has no data (ARM hasn't run, or no rules meet thresholds), each consumer falls back to current behavior
- Two new prompt templates, not modifications to existing ones
- Cross-clause extraction fields are clearly marked as "suggested" with lower default confidence than primary fields

---

## Section 5: Pipeline Orchestration and Phasing

### Two-Pass Execution Flow

```
PASS 1 — Foundation (structural + clustering)
  1.  Parse PDFs (pdf_parser — unchanged)
  2.  Structural decompose (nupunkt_chunker — NEW)
      → blocks → sentences → chunks with hierarchy IDs
  3.  KeyBERT score chunks (keybert_scorer — NEW)
      → clause types + keyphrases per chunk
  4.  Augment chunk text with KeyBERT prefix
  5.  Embed augmented text (embedder — modified input)
  6.  Macro cluster with semi-supervised UMAP (modified)
  7.  Micro cluster per domain with semi-supervised UMAP (modified)
  8.  Topic merge (unchanged)
  9.  Assign with KeyBERT confidence bias (modified)

PASS 2 — Enrichment (relationships + downstream)
  10. Build clause presence matrix from assignments
  11. ARM: mine clause rules (arm_miner — NEW)
  12. ARM: mine field correlations (field_arm — NEW)
  13. Populate relationship layer (NEW)
  14. Field discovery with ARM context (modified)
  15. Extraction with cross-clause context (modified)
  16. Knowledge graph with ARM edges (modified)
```

### Implementation Phases

| Phase                           | Scope                                                                          | New/Modified Files                                                                                                                                                                                        | Milestone                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Phase 1: nupunkt chunking       | Structural decomposition, block/sentence metadata, graceful fallback           | `core/nupunkt_chunker.py` (new), `core/chunker.py` (add entry point), `core/store.py` (schema), `config.py`                                                                                               | Chunks carry `block_id` + `sentence_ids`, existing tests pass                  |
| Phase 2: KeyBERT context        | Scoring, text augmentation, semi-supervised UMAP, confidence bias              | `core/keybert_scorer.py` (new), `core/cuml_wrappers.py` (target param), `core/cluster_assigner.py` (bias), `config.py`                                                                                    | Clustering quality improves on eval harness (silhouette, coherence)            |
| Phase 3: ARM mining             | Clause-level and field-level FP-Growth, relationship layer, DuckDB persistence | `core/arm/__init__.py`, `core/arm/arm_miner.py`, `core/arm/field_arm_miner.py`, `core/arm/configs.py` (all new), `core/relationship_layer.py` (new), `core/store.py` (schema), `pyproject.toml` (mlxtend) | Rules mined and queryable from existing pipeline outputs                       |
| Phase 4: Downstream integration | Field discovery, extraction, and KG consume ARM relationships                  | `core/field_discovery.py` (modified), `core/extractor.py` (modified), `core/context_graph.py` (modified), `prompts/field_discovery_arm.txt` (new), `prompts/field_extraction_arm_conditioned.txt` (new)   | End-to-end: PDF → enriched clusters → relationships → context-aware extraction |

### Orchestrator Changes

`cluster_pipeline.py` and `evoc_pipeline.py` each gain a `run_enrichment_pass()` function called after clustering completes. Controlled by a new config flag `ENABLE_ARM_ENRICHMENT` (default `True`). When `False`, Pass 2 steps 10-13 are skipped and downstream consumers use current behavior.

### Artifact Persistence

ARM outputs are part of the run artifacts under the existing layout:

```
data/artifacts/<account_id>/<run_id>/
  ├── macro_model/          (existing)
  ├── micro_models/         (existing)
  ├── embeddings.npz        (existing)
  ├── run_manifest.json     (existing — extended with ARM metadata)
  ├── arm_rules.json        (new — clause + field rules)
  └── term_packages.json    (new — frequent clause bundles)
```

### Validation

Each phase adds checks to `scripts/validate_pipeline.py`:

- Phase 1: `test_nupunkt_chunker()` — block/sentence decomposition on sample text
- Phase 2: `test_keybert_scorer()` — fingerprint loading, scoring, augmentation
- Phase 3: `test_arm_miner()` — FP-Growth on synthetic clause matrix
- Phase 4: `test_relationship_integration()` — relationship layer queries return results

---

## Dependencies

### New Python Packages

| Package   | Version   | Purpose                                                                     |
| --------- | --------- | --------------------------------------------------------------------------- |
| `nupunkt` | >= 0.6.0  | Legal-text-aware sentence/paragraph tokenization                            |
| `mlxtend` | >= 0.23.0 | FP-Growth association rule mining                                           |
| `keybert` | >= 0.8.0  | Keyphrase extraction (used by keybert_scorer for fingerprint compatibility) |

### Data Files

| File                       | Source                  | Destination                            |
| -------------------------- | ----------------------- | -------------------------------------- |
| `clause_fingerprints.json` | keyBERTClause `models/` | `data/models/clause_fingerprints.json` |

---

## New Config Variables Summary

| Variable                           | Default                                | Phase | Purpose                                |
| ---------------------------------- | -------------------------------------- | ----- | -------------------------------------- |
| `KEYBERT_FINGERPRINTS_PATH`        | `data/models/clause_fingerprints.json` | 2     | Path to fingerprints JSON              |
| `KEYBERT_CLASSIFICATION_THRESHOLD` | `0.3`                                  | 2     | Minimum similarity to assign a label   |
| `KEYBERT_UMAP_TARGET_WEIGHT`       | `0.3`                                  | 2     | Semi-supervised UMAP balance           |
| `KEYBERT_PRIOR_WEIGHT`             | `0.15`                                 | 2     | Confidence bias strength               |
| `KEYBERT_AUGMENT_TOP_K`            | `3`                                    | 2     | Keyphrases in text augmentation prefix |
| `ENABLE_ARM_ENRICHMENT`            | `True`                                 | 3     | Toggle Pass 2 enrichment               |
| `ARM_MIN_SUPPORT`                  | `0.20`                                 | 3     | Clause-level ARM minimum support       |
| `ARM_MIN_CONFIDENCE`               | `0.7`                                  | 3     | Clause-level ARM minimum confidence    |
| `ARM_MIN_LIFT`                     | `1.8`                                  | 3     | Clause-level ARM minimum lift          |
| `FIELD_ARM_MIN_SUPPORT`            | `0.05`                                 | 3     | Field-level ARM minimum support        |
| `FIELD_ARM_MIN_CONFIDENCE`         | `0.5`                                  | 3     | Field-level ARM minimum confidence     |
| `FIELD_ARM_MIN_LIFT`               | `1.5`                                  | 3     | Field-level ARM minimum lift           |

---

## DuckDB Schema Additions

All additions via idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` / `CREATE TABLE IF NOT EXISTS`.

### Modified Tables

- `chunks` — add `block_id TEXT`, `sentence_ids TEXT`, `keybert_scores TEXT`, `augmented_text TEXT`
- `clusters` — add `dominant_clause_type TEXT`, `clause_type_distribution TEXT`

### New Tables

- `arm_rules` — `rule_id TEXT PRIMARY KEY`, `antecedent TEXT` (JSON), `consequent TEXT` (JSON), `rule_type TEXT`, `support REAL`, `confidence REAL`, `lift REAL`, `cluster_id TEXT`, `pipeline_run_id TEXT`
- `arm_term_packages` — `package_id TEXT PRIMARY KEY`, `clause_types TEXT` (JSON), `support REAL`, `size INTEGER`, `cluster_id TEXT`, `pipeline_run_id TEXT`

---

## Graceful Degradation

Every new capability degrades independently:

| Missing Component                     | Behavior                                                    |
| ------------------------------------- | ----------------------------------------------------------- |
| nupunkt not installed                 | Falls back to regex sentence splitting (current behavior)   |
| Fingerprints file missing             | No augmentation, unsupervised UMAP, no confidence bias      |
| mlxtend not installed                 | ARM mining skipped, RelationshipLayer returns empty results |
| No cluster assignments yet            | ARM skipped, downstream consumers use current behavior      |
| ARM returns no rules above thresholds | Downstream consumers use current behavior                   |
