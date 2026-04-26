# Clustering V2: Agreement Intelligence Platform — Engineering Design Document

---

## Glossary

| Term | Definition |
|------|------------|
| Ontology | A structured, formal representation of knowledge within a domain — defining the types of entities that exist, their properties, and the relationships between them. In this system, the ontology is the organized catalog of clause types, their fields, hierarchical relationships, and intent mappings, built automatically from clusters rather than manually curated. It answers: "What kinds of clauses exist, what do they contain, and how do they relate?" |
| Cluster | A group of semantically similar chunks discovered by BERTopic (UMAP + HDBSCAN). Macro clusters group agreements by domain (e.g., SaaS, NDA); micro clusters group clause patterns within a domain. |
| ClauseType | A named ontology entity derived from a cluster (e.g., "Payment Terms"). Includes label, keywords, fields, and confidence score. |
| Field | A structured attribute extracted from a clause type (e.g., `payment_days` ). Can be universal (extracted from all agreements) or domain-specific (extracted only within a cluster). |
| Derived Field | A computed/composite field defined from base fields (e.g., `last_opt_out_date = expiration - notice_period` ). |
| Intent | An abstraction of *why* a clause exists and what tradeoff it encodes (e.g., cash flow optimization vs vendor friction). |
| Intent Dimension | A measurable axis of a tradeoff (e.g., cash_flow, risk, vendor_friction). |
| Feature Store | Persistent storage of pre-computed signals (cluster stats, org preferences, field distributions) for fast recommendations. |
| Reasoning Trace | Structured evidence chain explaining why a recommendation was made — never LLM-generated, always from stored data. |
| Dense Embedding | Single-vector representation of a chunk (MiniLM-384). Used for clustering and ANN search. |
| Cross-Encoder Reranker | A model that takes raw (query, document) text pairs as input and scores relevance via full cross-attention. Most accurate reranking method; zero storage overhead since nothing is precomputed. Replaces ColBERT for top-50 candidate reranking in this system. |
| Nomic Embed v1.5 | A 768-dim embedding model with 8K-token context window and Matryoshka representation support (dims can be truncated to 256/384/512). Candidate upgrade from MiniLM for Phase 2+. |
| Matryoshka Representation | An embedding training technique where a single model produces vectors that can be truncated to smaller dimensions (e.g., 768 → 384) with graceful quality degradation. Enables using the same model for both fast/compact clustering and high-precision retrieval. |
| Contrastive Learning | Training an embedding model by pulling similar pairs closer and pushing dissimilar pairs apart in embedding space. Used here (Phase 3+) to fine-tune embeddings on legal-specific semantics from accumulated extraction pairs. |
| c-TF-IDF | Class-based TF-IDF used by BERTopic for topic representation and keyword extraction. |
| ANN | Approximate Nearest Neighbor search (e.g., FAISS). Fast similarity lookup over large embedding indices. |
| HyDE | Hypothetical Document Embedding — generate a hypothetical answer to improve retrieval query quality. |
| Cold Start | The challenge of making recommendations for new users/organizations with no interaction history. |
| Macro Cluster | A high-level agreement domain (e.g., SaaS, NDA, Employment) formed by embedding-based summary clustering (BERTopic on document-level summary embeddings), with LLM used only to label the resulting clusters. |
| Micro Cluster | A clause-level pattern within a macro cluster, discovered by BERTopic on semantic embeddings. |
| Cluster-Conditioned Extraction | Extraction prompts parameterized with the cluster's label, known fields, and representative examples for higher precision. |
| Example Set | A representative sample (~20–50 docs) from a cluster used for initial extraction validation before scaling to the full corpus. |
| Concept Profile | The set of clause types, fields, and intent dimensions associated with a cluster — its "agreement genome." |
| BM25 (Best Matching 25) | A probabilistic sparse retrieval algorithm that scores documents by exact term frequency, inverse document frequency, and document length normalization. Unlike dense embeddings, BM25 excels at matching specific keywords, defined terms, statute references, and named entities. Used in this system both within BERTopic's c-TF-IDF weighting and as the sparse retrieval stage in the hybrid retrieval pipeline. |
| Sparse Retrieval | Retrieval based on exact lexical (keyword) matching rather than semantic similarity. BM25 is the dominant sparse retrieval method. Complements dense retrieval by catching exact terms that embeddings may miss. |

---

## 1. System Context

### 1.1 Current Architecture (V1)

```
┌─────────────────────────────┐
│  Service Bus (clustering-core) │
└──────────────┬──────────────┘
               ▼
┌──────────────────────────────────┐
│  apr-agreement-clustering        │
│  ├─ ClusteringExecutor           │
│  ├─ BayesianOptimizer / GridSearch│
│  ├─ BERTopicClusterer            │
│  │   ├─ UMAP DimensionalityReducer│
│  │   ├─ HDBSCAN DensityClusterer │
│  │   ├─ CountVectorizer (stopword filtering) │
│  │   ├─ ClassTfidfTransformer (BM25) │
│  │   └─ Outlier reduction (c-TF-IDF, threshold 0.3) │
│  └─ Azure Blob artifact storage  │
└──────────────┬───────────────────┘
               │ (gRPC)
┌──────────────▼───────────────────┐
│  AIDB Service                     │
│  ├─ FieldSummaryEmbeddingsService │
│  │   (minilm-v3-384)             │
│  ├─ ClusteringWriteAPI           │
│  └─ SnippetRetrievalService      │
└──────────────┬───────────────────┘
               │ (Service Bus: schema-gen)
┌──────────────▼───────────────────┐
│  apr-fields-schematizer           │
│  ├─ SchematizeHandler            │
│  ├─ CochranFiniteSelector        │
│  ├─ HyDE query generation        │
│  ├─ AIDB snippet selection       │
│  └─ GenericOpenAI extraction     │
└──────────────────────────────────┘
               │ (gRPC)
┌──────────────▼───────────────────┐
│  AIFlow Service                   │
│  ├─ GenericOpenAI                │
│  └─ EmbeddingService             │
└──────────────────────────────────┘

External: Redis (job status), Azure Blob (models/artifacts), Azure Service Bus
```

### 1.2 Current Pipeline Parameters

**Bayesian Optimization Search Space:**
* UMAP: `n_neighbors` (15–50),  `n_components` (5–15),  `min_dist` (0.0–0.1)
* HDBSCAN: `min_cluster_size` (5–20),  `min_samples` (3–12),  `epsilon` (0.0–0.15)

**Quality Thresholds:**
* Minimum clusters: 10
* Max dominance ratio: 0.2 (no single cluster > 20%)
* Max outlier ratio: 0.1 (< 10% unclustered)
* Keyword diversity threshold: 0.5

**Optimization:** 30 trials, 5 startup trials (Optuna TPE sampler)

---

## 2. V2 Target Architecture

### 2.1 Data Contract: Migration Path

**Critical architectural decision:** The current system clusters **field summaries** (one embedding per agreement field). The V2 target architecture operates on **clause-level semantic chunks** (multiple embeddings per agreement).

This is not a minor change. It affects embedding dimensionality, cluster semantics, extraction quality, and every downstream layer. The migration is phased:

| Phase | Input Unit | Source | Impact |
|-------|-----------|--------|--------|
| V1 (current) | Field summary | AIDB `FieldSummaryEmbeddingsService` | 1 embedding per field per agreement |
| V2 Phase 1–2 | Field summary (unchanged) | Same service | No re-indexing; ontology + compliance built on existing embeddings |
| V2 Phase 3+ | Semantic chunks (paragraphs) | New chunking pipeline + AIDB embedding service | Re-embedding required; cluster re-training; ontology re-labeling |

**Phase 1–2 explicitly operate on the existing field-summary data contract.** No migration is required. The ontology, compliance, and recommendation layers are designed to work with the current embedding granularity.

**Phase 3+ migration plan (when chunk-level processing is introduced):**

1. **Dual-write period:** New agreements get both field-summary and chunk-level embeddings. Existing agreements are backfilled in batches.
2. **Shadow clustering:** Run chunk-level BERTopic in parallel with field-summary BERTopic. Compare cluster quality metrics (purity, stability, coherence) on gold set.
3. **Ontology re-mapping:** Re-label chunk-level clusters using the same LLM pipeline. Map new cluster IDs to existing `ClauseType` entities where possible; flag new/orphaned types for human review.
4. **Cutover gate:** Chunk-level pipeline must meet or exceed field-summary pipeline on all evaluation framework thresholds before cutover.
5. **Rollback:** Field-summary pipeline remains operational for 90 days post-cutover.

**Re-indexing cost estimate:**
* 10M agreements × ~8 chunks/agreement × 384-dim × 4 bytes = ~120 GB embedding storage
* Embedding compute: ~$500–$1, 000 (batch MiniLM inference on GPU)
* Cluster re-training: < 1 hour on existing infrastructure
* Ontology re-labeling: ~$50 (50 clusters × GPT-4o-mini)

### 2.2 Architecture Diagram

```
┌─────────────────────────────────────────────┐
│              DocuSign Agreements              │
└──────────────────────┬──────────────────────┘
                       ▼
              Parsing + Segmentation
                       ▼
              ┌─────────────────┐
              │ Embedding Layer  │
              │ (MiniLM-384)    │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Clustering Layer │
              │ (BERTopic)      │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Ontology Layer   │
              │ (Cluster→Clause) │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Intent Layer     │
              │ (Clause→Intent)  │
              └────────┬────────┘
                       ▼
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
┌────────┐    ┌──────────────┐    ┌────────────┐
│Policy  │    │Recommendation│    │Personalizn │
│Engine  │    │Engine        │    │Engine      │
└────┬───┘    └──────┬───────┘    └─────┬──────┘
     │               │                  │
     └───────────────┬┘                 │
                     ▼                  │
              ┌─────────────────┐       │
              │ Reasoning Layer  │◄──────┘
              │ (Traces + Expl.) │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Feedback Loop    │
              │ (Learn + Update) │
              └─────────────────┘
```

---

## 3. Clustering Layer (Enhanced)

### 3.1 Two-Level Clustering: Macro → Micro

Pure semantic-embedding clustering produces flat topic groups where structurally different agreement types can merge (e.g., an NDA indemnity clause and a SaaS indemnity clause land in the same cluster because they sound similar, but they belong to different legal domains with different field schemas).

To fix this, clustering operates in two phases:

**Phase A — Macro Clusters (Agreement Domain)**

Group agreements into high-level domains by running **BERTopic on document-level summary embeddings** — not per-document LLM classification:

1. **Build summary embeddings:** For each agreement, embed a summary string (title + first ~500 tokens of body text) using MiniLM-384.
2. **Cluster summaries:** Run UMAP + HDBSCAN on these summary-level embeddings to discover macro clusters. This typically produces 3–10 coarse domain groups.
3. **Label clusters via LLM:** Call GPT-4o-mini once per macro cluster (5–10 LLM calls total, not per-document) to produce a domain label and description from the cluster's keywords + representative docs.
4. **Enrich with metadata signals (optional):** Where available, use AIDB metadata (agreement type, industry) and structural features (section count, page count) as auxiliary signals. These can be incorporated as additional features before UMAP, or used to validate/merge clusters post-hoc.

| Signal | Source | Role |
|--------|--------|------|
| Summary embedding | MiniLM on title + opening text | **Primary clustering signal** |
| Agreement type | AIDB metadata (if available) | Auxiliary feature / post-hoc validation |
| Industry vertical | Org metadata (if available) | Auxiliary feature |
| Structural features | Document analysis (section count, page count) | Auxiliary feature |
| Regulatory context | Metadata or LLM inference | Post-hoc enrichment |

**Why embedding-based macro clustering instead of per-doc LLM classification:**
* **Cost:** 5–10 LLM calls to label clusters vs 100–10, 000 LLM calls to classify individual documents. At scale, per-doc classification is prohibitively expensive.
* **Speed:** Embedding + BERTopic runs locally in seconds. Per-doc LLM calls are I/O-bound and take minutes for large corpora.
* **Consistency:** BERTopic produces deterministic cluster boundaries (with fixed seeds). LLM classification can produce inconsistent domain assignments for borderline documents.
* **Discoverability:** Clustering can discover unexpected domains that weren't in a predefined taxonomy. LLM classification is constrained to a fixed label set.

Macro clustering uses BERTopic on summary embeddings (same proven approach as micro clustering, just at coarser document-level granularity). Each macro cluster represents an **agreement domain** — a family of structurally and functionally similar contracts.

**Phase B — Micro Clusters (Clause Patterns within Domain)**

Within each macro cluster, run BERTopic (UMAP + HDBSCAN) on semantic embeddings to discover clause-level patterns. This is the existing pipeline, scoped to a domain:

```
Macro: SaaS Agreements (500 docs)
  ↓ BERTopic within this domain
Micro clusters:
  - Payment Terms (cluster 0)
  - Liability Limitation (cluster 1)
  - Auto-Renewal (cluster 2)
  - Data Processing (cluster 3)
  - SLA / Uptime (cluster 4)
  ...
```

**Why this matters:**
* Indemnity in an NDA vs indemnity in a SaaS agreement have different fields, different norms, and different policy implications. Flat clustering conflates them.
* Macro→micro gives stable, meaningful domains and enables **domain-conditioned extraction** (Section 4.5).
* Users can also provide natural-language taxonomy hints (e.g., "procurement", "supply chain") that bias macro cluster formation.

**Configuration:** The signals and weights used for macro clustering are **admin-configurable per organization** via the `OntologyService` . Organizations with strong metadata can incorporate structural signals as auxiliary features alongside summary embeddings; those without metadata rely purely on the embedding-based clustering. Users can also provide natural-language taxonomy hints (e.g., "procurement", "supply chain") that bias macro cluster formation via guided topic modeling.

**Minimum doc thresholds:**
* Macro cluster assignment: No minimum (single doc can be classified into a domain)
* Micro clustering (BERTopic): Target ~100 docs per macro cluster for stable micro clusters. Below this, use the example-set pattern (Section 17, Phase 1) with LLM-only field discovery instead of statistical clustering.
* Gold set for evaluation: 500 labeled summaries across 10 accounts (unchanged)

### 3.2 Embedding Strategy

**Primary: Dense Embeddings (MiniLM-384)**

Continue using existing `minilm-v3-384` embeddings via AIDB. These provide the scale + speed needed for clustering millions of chunks.

**Why MiniLM remains sufficient for clustering:**
* Clustering needs global semantic signal, not token-level precision
* 384-dim vectors are compact and fast for ANN search
* Already integrated with AIDB infrastructure

**Consideration: LegalBERT for precision tasks (not clustering)**

LegalBERT better captures legal nuance ("indemnify" vs "hold harmless, " "shall" vs "may"), but is slower and heavier. Use it selectively:

| Stage | Model | Rationale |
|-------|-------|-----------|
| Clustering + ANN | MiniLM-384 | Speed, scale, good-enough semantics |
| Reranking / precision retrieval | LegalBERT or ColBERT | Fine-grained legal nuance |
| Contrastive fine-tuning (later) | Custom model | Domain-specific learning |

**Hybrid retrieval architecture (Phase 5+):**

```
Query
  ↓
├─ Dense: MiniLM embedding → ANN search → Top-100
├─ Sparse: BM25 (inverted index) → Top-100
  ↓
RRF merge → Top-100 combined candidates
  ↓
Cluster-aware filtering (restrict to relevant topic clusters)
  ↓
ColBERT-style reranking → Top-5 precise matches
  ↓
Intent-aligned scoring
  ↓
Final recommendation
```

### 3.3 On Mean Pooling for Document Representation

The current system embeds field summaries directly. For future scenarios involving chunk-level processing:

**Mean pooling of chunk embeddings is an acceptable baseline** but has known weaknesses for multi-topic legal documents:

* **Topic dilution:** Different clauses average out → blurry embedding
* **Signal loss:** Rare but critical clauses (force majeure) get averaged away
* **Structure blindness:** Ignores which clauses exist and their relative importance

**Recommended approach for document-level representation:**

```
Primary:   topic_distribution_vector = [P(topic_1), P(topic_2), ..., P(topic_k)]
Secondary: mean_embedding (as auxiliary feature)
Optional:  key_chunk_embeddings (for retrieval)
```

Topic distribution preserves multi-topic structure, aligns with legal clause reality, and is more interpretable.

### 3.4 BERTopic Enhancements

**Current pipeline preserved:**

```
Pre-computed embeddings (AIDB)
    ↓
UMAP dimensionality reduction
    ↓
HDBSCAN density clustering
    ↓
c-TF-IDF + BM25 representation
    ↓
Outlier reduction (cosine threshold 0.3)
    ↓
Optional: GPT-based topic naming
```

**Why c-TF-IDF uses BM25 weighting:**

BERTopic's `ClassTfidfTransformer` applies BM25-style sub-linear term-frequency saturation ( `tf * (k1 + 1) / (tf + k1)` ) instead of raw counts. This prevents a single high-frequency term (e.g., "agreement", "party") from dominating a topic's representation. In legal corpora this is especially important because boilerplate phrasing appears in nearly every document — BM25 weighting ensures that **distinguishing** terms (e.g., "indemnification", "force majeure", "auto-renewal") surface as the topic's representative keywords rather than generic legalese. The `b` parameter (document length normalization) further corrects for the wide variance in clause lengths across agreement types.

**V2 additions:**

1. **Cluster labeling pipeline:** Automated LLM labeling of every cluster → `ClauseType` entity
2. **Cluster quality scoring:** Per-cluster coherence metrics (silhouette, c-TF-IDF diversity, intra-cluster cosine similarity)
3. **Cluster versioning:** Track cluster evolution across re-runs; detect merges, splits, and drift
4. **Incremental cluster assignment:** New documents assigned to existing clusters without full re-clustering (via nearest-centroid or c-TF-IDF matching)

**Known BERTopic limitations to address:**

| Issue | Impact | Mitigation |
|-------|--------|------------|
| Non-deterministic (UMAP + HDBSCAN) | Different runs → different clusters | Fix random seeds; version cluster state; track cluster lineage |
| Over-clustering | Too many fine-grained topics | Merge clusters with high inter-cluster similarity |
| Mixed clusters | Payment + late fees in one cluster | Intra-cluster splitting using sub-clustering (Phase 5+) |
| Wording sensitivity | Small differences → different clusters | Handled today by HDBSCAN density; later improve via contrastive learning |

### 3.5 Clause-Level Semantic Embeddings (P0 — Phase 1–2)

> **[V2] Rearchitect Input Signal: Clause-Level Semantic Embeddings and Field Processor Performance Improvements**
>
> **This is a P0 critical implementation item, not a spike.** The architectural shift from field-summary embeddings to clause-level embeddings is fundamental to V2's value proposition and must be implemented in Phase 1–2.

#### 3.5.1 Problem Statement

The current V1 pipeline embeds **field summaries** (one aggregated embedding per agreement). This has fundamental limitations:

| Issue | Impact | Example |
|-------|--------|--------|
| **Semantic dilution** | Different clause types averaged together lose distinguishing signals | Liability clause + Payment clause in same doc → blurry embedding |
| **Critical term loss** | Rare but important clauses get averaged away | Force majeure appears in 5% of text but is decision-critical |
| **Structure blindness** | No awareness of which clauses exist or their relative positions | Can't distinguish "has liability cap" from "mentions liability" |
| **Granularity mismatch** | Agreement-level embedding used for clause-level clustering | Micro-clusters conflate clauses from different semantic regions |

#### 3.5.2 Target Architecture: Per-Clause Embeddings

```
V1 (Current):   Agreement → Field Summary → 1 Embedding → Clustering
                                         ↓
                              (semantic dilution)

V2 (Target):    Agreement → Clause Chunks → N Embeddings → Clustering
                              ↓               ↓
                         (structure)    (per-clause signals)
```

**Per-clause embedding preserves:**
* Clause-type-specific semantics (liability vs payment vs termination)
* Position and structural context (preamble vs operative vs schedule)
* Clause-level similarity for micro-clustering
* Field extraction context (each clause carries its own embedding)

#### 3.5.3 Embedding Strategy Options

| Strategy | Description | Pros | Cons | Recommended Phase |
|----------|-------------|------|------|------------------|
| **Direct clause embedding** | Embed each clause chunk independently | Simple, parallelizable | Short clauses lack context | Phase 2 (baseline) |
| **Contextualized embedding** | Embed clause + N surrounding chunks | Better context for short clauses | Slightly slower | Phase 2–3 |
| **Hierarchical embedding** | Section → clause → sentence embeddings | Rich multi-scale representation | Complex, higher storage | Phase 4+ |
| **Instruction-tuned embedding** | Use Nomic/E5 with task prefix | Better domain alignment | Model dependency | Phase 3 |

**Recommended implementation:**

```python
class ClauseLevelEmbedder:
    """
    Clause-level embedding with optional context window.

    Key design decisions:
    1. Each clause gets its own embedding (not averaged)
    2. Optional context window for short clauses
    3. Section heading included for structural signal
    4. Normalized embeddings for cosine similarity
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", context_window: int = 1):
        self.model = SentenceTransformer(model_name)
        self.context_window = context_window

    def embed_clauses(self, chunks: list[dict]) -> np.ndarray:
        """Embed each clause with structural context.

        Args:
            chunks: [{chunk_id, text, section_heading, position}, ...]

        Returns:
            (n_chunks, embedding_dim) matrix
        """
        texts = []
        for i, chunk in enumerate(chunks):
            # Include section heading for structural signal
            heading = chunk.get("section_heading", "")
            prefix = f"[{heading}] " if heading else ""

            # Add context window for short clauses
            if len(chunk["text"]) < 100 and self.context_window > 0:
                context_before = " ".join(
                    c["text"] for c in chunks[max(0, i-self.context_window):i]
                )
                text = f"{prefix}{context_before} {chunk['text']}".strip()
            else:
                text = f"{prefix}{chunk['text']}"

            texts.append(text)

        return self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=32
        )
```

#### 3.5.4 Field Processor Performance Improvements

This implementation also addresses extraction pipeline inefficiencies:

| Current Issue | Impact | Improvement |
|--------------|--------|-------------|
| Sequential LLM calls | 1 call per field per doc = O(fields × docs) | Batch extraction: 10–20 snippets per call |
| Repeated prompt compilation | Same prompt rebuilt per call | Prompt caching by cluster type |
| No early termination | Extract all fields even for high-confidence | Skip LLM for stable fields (regex fallback) |
| Blocking I/O | Waits for each LLM response | Async parallel with rate limiting |

**Performance targets:**

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| LLM calls per cluster | N × F (docs × fields) | N/10 × F (batched) | 10x reduction |
| Extraction latency (100 docs) | ~5 min | ~30 sec | 10x faster |
| Prompt tokens per extraction | ~2000 | ~1200 | 40% reduction |
| Stable field extraction | LLM call | Regex | 100% token savings |

#### 3.5.5 Migration Path: Field Summaries → Clause Embeddings

This is a **breaking change** to the data contract. Migration requires careful planning:

| Phase | Embedding Granularity | Migration Step |
|-------|----------------------|----------------|
| Phase 1–2 | Field summaries (V1) | No change; build ontology on existing embeddings |
| Phase 1–2 | Clause embeddings (shadow) | Dual-write: store both; compare clustering quality |
| Phase 2–3 | Clause embeddings (primary) | Cutover when validation shows >5% cluster purity improvement |
| Phase 3+ | Clause embeddings only | Deprecate field summary embeddings; reclaim storage |

**Re-indexing cost (10M agreements):**
* Embedding compute: ~$500–1,000 (batch MiniLM on GPU)
* Storage delta: +~80 GB (8 chunks/doc × 10M × 1.5 KB/embedding − existing summary embeddings)
* Cluster re-training: <1 hour
* Ontology re-labeling: ~$50 (cluster labels via LLM)

#### 3.5.6 Implementation Timeline & Success Criteria

| Phase | Milestone | Success Criteria |
|-------|-----------|------------------|
| Phase 1 (Hackathon) | Prototype `ClauseLevelEmbedder` | Working demo with side-by-side comparison |
| Phase 1 (Week 3-4) | Production implementation | Integrated into clustering pipeline |
| Phase 2 (Week 5-8) | Migration complete | All new agreements use clause-level embeddings |
| Phase 2 (Week 9-12) | Backfill historical | Existing agreements re-embedded |

| Metric | Baseline (Field Summary) | Target (Clause-Level) | Measurement |
|--------|-------------------------|----------------------|-------------|
| Cluster purity | Baseline | +5% | Manual review of 100 clusters |
| Extraction F1 | Baseline | +3% | Gold-labeled extraction set |
| Extraction latency | Baseline | -50% | End-to-end timing |
| Token cost | Baseline | -30% | LLM API metrics |

**Implementation is complete when:** All 4 metrics show improvement AND migration is complete for >95% of agreements.

---

## 4. Ontology Layer

### 4.1 Purpose

Convert BERTopic clusters into a structured, queryable ontology of clause types, fields, and relationships.

**Key insight:** BERTopic clusters are already a **latent ontology**. The ontology layer formalizes and enriches them.

### 4.2 Data Model

```
ClauseType {
    clause_type_id: string
    label: string                    // "Payment Terms"
    source_topic_id: int             // BERTopic topic_id
    keywords: list[string]
    representative_chunks: list[string]
    fields: list[Field]
    confidence: float
    parent_clause_type_id: string?   // for hierarchy (Phase 5+)
    created_at: timestamp
    version: int
}

Field {
    field_id: string
    name: string                     // "payment_days"
    type: enum[string, int, float, bool, date, enum]
    clause_type_id: string
    coverage_score: float            // % of cluster where field appears
    value_distribution: dict         // {"30": 0.4, "45": 0.45, "60": 0.15}
}

AgreementClause {
    agreement_id: string
    clause_type_id: string
    source_chunks: list[chunk_id]
    extracted_fields: dict           // {"payment_days": 30}
    confidence: float
}
```

### 4.3 Universal vs Domain-Specific Fields

Not all fields belong to a specific cluster. Some fields appear across every agreement type and should be extracted globally, independent of cluster assignment.

**Universal fields (extracted for ALL agreements, regardless of cluster):**

| Field | Type | Examples |
|-------|------|----------|
| Contracting parties | string[] | "Acme Corp", "Widget Inc" |
| Effective date | date | 2025-01-15 |
| Governing law / jurisdiction | string | "State of California" |
| Agreement type | enum | SaaS, NDA, Employment, Vendor |
| Term / duration | string | "2 years" |
| Termination notice period | string | "30 days" |

**Domain-specific fields (extracted only for agreements in the relevant cluster):**

| Field | Cluster | Examples |
|-------|---------|----------|
| payment_days | Payment Terms | 30, 45, 60 |
| liability_cap | Liability Limitation | "2x fees", "unlimited" |
| auto_renew | Auto-Renewal | true, false |
| data_retention_period | Data Processing | "3 years", "duration of agreement" |
| uptime_sla | SLA / Uptime | "99.9%", "99.95%" |

**Architecture implication:** Universal extraction runs on every agreement as a parallel pipeline (can use current standard extraction or a dedicated universal prompt). Domain-specific extraction runs only after cluster assignment, using cluster-conditioned prompts (Section 4.5). This separation:
* Avoids extracting payment_days from an NDA (wasted tokens, hallucination risk)
* Ensures baseline fields are always available even if cluster assignment fails
* Reduces per-agreement LLM cost by ~30–50% (fewer irrelevant fields attempted)

### 4.4 Ontology Construction Pipeline

```
Step 0: Universal Extraction (parallel, all agreements)
    - Extract universal fields using a domain-agnostic prompt
    - Store in AgreementClause records with clause_type = "universal"

Step 1: Cluster Labeling
    For each BERTopic topic (micro cluster):
        - Sample 10–20 representative chunks
        - LLM prompt: "Identify the legal clause category"
        - Output: clause_type label + confidence

Step 2: Field Discovery (domain-specific)
    For each ClauseType:
        - Use existing SchematizeHandler pipeline
        - HyDE query → hybrid snippet retrieval (dense + BM25) → LLM extraction
        - Cluster-conditioned prompt (see Section 4.5)
        - Compute field coverage and value distributions

Step 3: Agreement-Clause Mapping
    For each agreement:
        - Map paragraphs to their assigned topics
        - Create AgreementClause records
        - Merge universal + domain-specific fields
```

### 4.4.1 RLM-Based Field Discovery (DSPy) — Hackathon Validated

> **[V2] RLM Field Discovery: Agentic code-writing approach to field identification**
>
> **Validated during hackathon on CUAD-510 corpus (20 clusters × 510 agreements).** Provides complementary coverage to Standard field discovery. Production recommendation: use Hybrid strategy (§4.4.2).

#### Problem Statement

Standard field discovery (§4.4 Step 2) samples up to 30 representative chunks and makes a single LLM call. This is fast (4.4s/cluster) and 100% reliable, but the fixed sample window can miss rare-but-important fields that appear only in specific document subsets.

#### RLM Architecture

RLM (Recursive Language Model) uses DSPy's code-generation REPL to let the LLM **programmatically explore** the entire chunk corpus rather than passively reading a fixed sample:

```
Standard:  30 chunks → 1 LLM call → field list
RLM:       All chunks → LLM writes Python → regex scans + strategic sampling
             → 3–6 batched llm_query() calls → consensus voting → deduplicated fields
```

**Implementation (`discover_fields_for_cluster_rlm()`):**

```python
class FieldDiscoverySig(dspy.Signature):
    """Given clause excerpts (chunks_json) belonging to a clause type,
    discover ALL recurring structured fields via programmatic exploration.

    STRATEGY:
    1. Parse chunks and print 5 diverse chunks in full
    2. Use llm_query_batched to send 3-6 batches to sub-LLM
    3. Aggregate results — keep fields appearing in 2+ batches
    4. Deduplicate by name similarity
    """
    cluster_label: str = dspy.InputField()
    chunks_json: str = dspy.InputField()
    max_fields: int = dspy.InputField()
    fields_json: str = dspy.OutputField()  # JSON list of {name, type, description, examples}

# Execution via DSPy CodeAct with sandboxed REPL
rlm = dspy.CodeAct(
    FieldDiscoverySig,
    tools=[llm_query_batched, print_chunks, regex_scan],
    max_iterations=RLM_MAX_ITERATIONS,  # default: 25
    max_llm_calls=RLM_MAX_LLM_CALLS,    # default: 50
)
```

**Key design decisions:**
1. **Sandboxed REPL:** LLM code runs in a restricted environment — no file I/O, no network, no imports beyond allowlist
2. **Consensus voting:** Fields must appear in ≥2 of 3–6 batched sub-LLM calls to survive
3. **JSON parsing fallback chain:** `json.loads()` → `ast.literal_eval()` → markdown fence extraction → regex — handles the variety of LLM output formats  
4. **Graceful degradation:** On any RLM failure, falls back to Standard discovery (critical for production reliability)
5. **Sub-LM routing:** Inner `llm_query()` calls can use a cheaper model than the outer RLM orchestrator

#### Benchmark Results: Standard vs RLM (20-Cluster Evaluation)

| Metric | Standard | RLM | Delta |
|--------|----------|-----|-------|
| **Avg fields/cluster** | 11.0 | 10.0 | RLM finds fewer but different fields |
| **Avg latency/cluster** | **4.4s** | 36.5s | 8.3× slower |
| **Failure rate** | **0%** | 5% (1/20 clusters) | RLM can timeout or fail to parse |
| **Reliability** | **100%** | 95% | Standard never fails |
| **Jaccard overlap** | — | 0.105 | Only ~10% field overlap between approaches |
| **Type agreement (shared fields)** | — | 32% | Approaches often disagree on data types |

**Per-cluster highlights (20 clusters on CUAD-510):**

| Cluster | Std Fields | RLM Fields | Jaccard | Notes |
|---------|-----------|------------|---------|-------|
| Franchise Agreement | 12 | 0 | 0.000 | RLM failure — timeout |
| Mutual Fund Servicing | 12 | 8 | 0.333 | Best overlap — well-structured cluster |
| Contract Term Definitions | 10 | 15 | 0.316 | RLM excels on structured terms |
| Force Majeure Clauses | 11 | 15 | 0.182 | RLM finds more niche fields |
| Export Control | 15 | 14 | 0.000 | Both find many fields, but entirely different ones |
| Section Headings Disclaimer | 8 | 2 | 0.000 | RLM struggles with thin clusters |

**Key insight:** The near-zero Jaccard overlap (mean 0.105) demonstrates that Standard and RLM discover **largely complementary** field sets. This is the core motivation for the Hybrid strategy.

---

### 4.4.2 Hybrid Field Discovery Strategy — Recommended for Production

> **[V2] Hybrid Field Discovery: Union of Standard + RLM with 4-tier merge**
>
> **Validated during hackathon.** Produces 22% more fields at highest confidence (0.751) with 100% reliability. Recommended as the default field discovery strategy for V2.

#### Architecture

```
Hybrid Discovery Pipeline:

Standard Discovery (parallel)          RLM Discovery (parallel)
├─ 30 chunks sampled                   ├─ All chunks available via REPL
├─ 1 LLM call                         ├─ 3-6 batched sub-LM calls
├─ 4.4s avg                           ├─ 36.5s avg (or timeout → fallback)
├─ 100% reliable                      ├─ 95% reliable
└─ ~11 fields                         └─ ~10 fields
         │                                      │
         └──────────────┬───────────────────────┘
                        ↓
              4-Tier Merge Algorithm
                        ↓
               ~13.4 fields (hybrid)
               0.751 avg confidence
               100% reliable
```

#### 4-Tier Merge Algorithm (`_merge_field_lists()`)

The merge uses a priority-ordered strategy that maximizes coverage while preserving attribution:

```
TIER 1: Exact Name Matches → source="both"           (HIGHEST SIGNAL)
├─ Condition: Both approaches found identical field name (e.g., "notice_period")
├─ Action: Pick richer definition (longer description), combine unique examples (up to 5)
├─ Signal: Strongest validation — independent discovery by different methods

TIER 2: Fuzzy Substring Matches → source="both_fuzzy" (SEMANTIC EQUIVALENCE)
├─ Condition: Bidirectional substring containment
│   e.g., "notice_period" ⊆ "termination_notice_period"
├─ Action: Use shorter canonical name, merge descriptions
├─ Signal: Same concept, different granularity

TIER 3: Standard-Unique Fields → source="standard"    (RELIABLE BASE)
├─ Remaining fields only found by Standard
├─ Rationale: 100% reliable, fast, good broad coverage

TIER 4: RLM-Unique Fields → source="rlm"              (COMPLEMENTARY DEPTH)
├─ Remaining fields only found by RLM
├─ Rationale: Deep REPL exploration finds fields Standard misses
```

**Final output:** Merged list sorted by tier priority, capped at `max_fields` (configurable, default 15), with source attribution on each field.

#### Benchmark Results: Hybrid (5-Cluster Evaluation with Extraction)

| Metric | Standard | RLM | **Hybrid** |
|--------|----------|-----|------------|
| **Avg fields/cluster** | 11.6 | 9.6 | **13.4 (+22%)** |
| **Avg confidence** | 0.745 | 0.738 | **0.751 (highest)** |
| **Fill rate** | 33.6% | 35.4% | **36.5%** |
| **Failure rate** | 0% | 20% | **0%** |
| **Avg latency** | **4.5s** | 2.2s* | 6.7s |

*RLM times reflect timeout/early-termination in some clusters

**Hybrid source breakdown (5-cluster aggregate, 67 total fields):**

| Source | Count | % | Interpretation |
|--------|-------|---|----------------|
| `both` (exact match) | 10 | 14.9% | High-confidence validated fields |
| `both_fuzzy` (semantic match) | 7 | 10.4% | Same concept, different naming |
| `standard` (unique) | 41 | 61.2% | Broad reliable base coverage |
| `rlm` (unique) | 9 | 13.4% | Complementary deep-dive fields |

#### `run_field_discovery()` Orchestrator

The pipeline orchestrator accepts a `use_rlm` parameter:

```python
def run_field_discovery(db_path, *, use_rlm: str = "no", config=None):
    """
    Args:
        use_rlm: "no" (Standard only), "yes" (RLM only), "hybrid" (recommended)
    """
```

| Mode | Behavior | Latency Budget | Recommendation |
|------|----------|---------------|----------------|
| `"no"` | Standard only | ~4.4s/cluster | Dev / fast iteration |
| `"yes"` | RLM only | ~36.5s/cluster | Research / comparison |
| `"hybrid"` | Standard + RLM → merge | ~40s/cluster | **Production default** |

#### Cost & Latency Considerations

| Factor | Standard | RLM | Hybrid |
|--------|----------|-----|--------|
| LLM calls per cluster | 1 | 5–15 (orchestrator + sub-LM batches) | 6–16 |
| Token cost per cluster | ~2K tokens | ~15K tokens | ~17K tokens |
| GPU time | Negligible | Negligible | Negligible |
| **Marginal cost (gpt-4o-mini)** | ~$0.001 | ~$0.008 | ~$0.009 |

At $0.009/cluster with ~50 clusters per corpus, hybrid field discovery adds **~$0.45 per full corpus run** — negligible relative to extraction costs (~$0.033/doc × 510 docs = ~$16.83).

---

### 4.4.2.1 Latent Briefing: Cross-Worker Memory Sharing for RLM

> **[V2] Latent Briefing memory layer for RLM field discovery**
>
> Adapted from Ramp Labs' Latent Briefing (April 2026) which demonstrated 65% worker-token reduction, +3% accuracy improvement, and 1.7s median compaction overhead on LongBench v2. Implemented as a DSPy-compatible Text Latent Briefing layer for our API-based (Azure OpenAI) setup.

#### Problem Statement

In standard RLM field discovery (§4.4.1), the orchestrator sends independent `llm_query()` calls to worker models. Each worker call is stateless — worker call 3 has no knowledge of what workers 1 and 2 already discovered. This leads to:

| Waste | Impact | Example |
|-------|--------|---------|
| **Redundant discovery** | Workers re-find the same fields | Batch 3 discovers `effective_date` that batch 1 already found |
| **No gap awareness** | Later batches can't focus on uncovered areas | All batches ask the same generic "list all fields" question |
| **Token explosion** | Passing full trajectory as text costs too much | 25 REPL iterations × 6 worker calls = 150+ calls, growing context |
| **Noise accumulation** | Dead-end reasoning pollutes worker context | Regex scan for phone numbers is irrelevant to field discovery |

#### How Latent Briefing Solves This

Ramp Labs' core insight: share memory at the **representation level**, not the text level. For API-based models, we implement this as **Text Latent Briefing** — using embedding similarity as a proxy for the attention-based scoring in the original KV cache approach.

```
Standard RLM (stateless workers):

  Orchestrator → llm_query("List fields from chunks 0-9")  → Worker A (no context)
  Orchestrator → llm_query("List fields from chunks 10-19") → Worker B (no context)
  Orchestrator → llm_query("List fields from chunks 20-29") → Worker C (no context)

RLM + Latent Briefing (memory-augmented workers):

  Orchestrator → llm_query("List fields from chunks 0-9")
      ↓ record Worker A reply in trajectory memory
  Orchestrator → compact trajectory for chunk 10-19 task
      → Worker B sees: [BRIEFING: "Already found: effective_date, governing_law..."] + task
      ↓ record Worker B reply in trajectory memory
  Orchestrator → compact trajectory for chunk 20-29 task (different briefing!)
      → Worker C sees: [BRIEFING: "Payment fields found. Focus on liability/IP..."] + task
```

#### Three Inference-Time Modifications (Adapted from Ramp Labs)

| Modification | Original (KV Cache) | Our Adaptation (Text-Level) |
|-------------|---------------------|----------------------------|
| **1. Task-guided scoring** | Score trajectory positions using worker's task query vectors in attention space | Score trajectory segments using cosine similarity between task-prompt embedding and segment embeddings |
| **2. Shared global mask** | Aggregate attention scores across all heads into one per-position mask | Single relevance score per segment (embedding cosine), no per-head decomposition needed |
| **3. MAD thresholding** | `median + τ × MAD` on attention scores for adaptive compression | Same formula on cosine similarity scores — adapts to actual score distribution |

#### Implementation Architecture

```python
class TextLatentBriefing:
    """Text-level Latent Briefing for API-based models."""

    def add_trajectory_segment(text, segment_type):
        # Record REPL iterations: code, output, worker replies, reasoning
        # Embed each segment using MiniLM for scoring

    def compact_for_task(task_prompt, tau) -> (briefing_text, stats):
        # 1. Embed the task prompt (task-guided query vectors)
        # 2. Cosine similarity: task_embed vs each segment_embed
        # 3. Apply segment-type boosts (worker_reply +0.1, observation -0.05)
        # 4. MAD threshold: keep positions above median + tau * MAD
        # 5. Concatenate surviving segments in trajectory order
        # Returns compressed briefing + compaction statistics

class RLMWithLatentBriefing(dspy.Module):
    """DSPy RLM wrapper with memory-augmented llm_query() calls."""

    def forward(**inputs) -> Prediction:
        # Each llm_query() call:
        #   1. Compacts trajectory for THIS specific task
        #   2. Prepends briefing to worker prompt
        #   3. Records worker reply in trajectory
        # Result includes briefing_stats for observability
```

#### Adaptive Tau (Ramp Labs Finding)

Optimal compression varies with task difficulty and trajectory length:

| Regime | τ Value | Rationale |
|--------|---------|-----------|
| Early iterations (< 5 segments) | Base τ × 1.0 | Preserve initial exploration context |
| Mid iterations (5–15 segments) | Base τ × 1.2 | Moderate filtering of accumulating noise |
| Late iterations (> 15 segments) | Base τ × 1.5 | Aggressive filtering of dead-end reasoning |
| Very long trajectories (> 20 segments) | Reduce × 0.8 | Preserve dispersed evidence |

**Ramp Labs' counterintuitive finding:** Harder tasks benefit from **more** aggressive compression. Speculative reasoning (dead-end hypotheses, wrong regex patterns) is noise that actively harms worker performance. Compression removes the orchestrator's wrong guesses, letting the worker focus on real signal.

#### Validated Behavior: Task-Guided Selective Retention

The same trajectory produces **different briefings** for different worker tasks:

```
Trajectory (10 segments from RLM field discovery):
  Seg 1: "Parsed 60 chunks..."               (observation)
  Seg 2: "Batch 1: effective_date, parties"    (worker_reply)
  Seg 3: "Batch 2: payment_terms, late_fee"    (worker_reply)
  Seg 4: "Batch 3: liability_cap, indemnity"   (worker_reply)
  Seg 5: "Batch 4: termination, force_majeure" (worker_reply)
  Seg 6: "Batch 5: confidentiality, ip_owner"  (worker_reply)
  ...

Task: "What payment/financial terms exist?"     → Retains Seg 3 (payment)
Task: "What termination/liability provisions?"  → Retains Seg 4 (liability)
Task: "What IP/confidentiality protections?"    → Retains Seg 6 (IP)
Task: "What date/duration fields?"              → Retains Seg 2 + Seg 3
```

With τ=2.0: **71–91% compression** while retaining task-relevant segments.

#### Infrastructure Decision: Text vs KV Cache

| Approach | When to Use | Our Setup |
|----------|-------------|-----------|
| **KV Cache Latent Briefing** | Self-hosted model with runtime KV access (vLLM, TGI) | Available — 2× A100 80GB, Qwen2.5-3B local, vLLM 0.7.3. Implemented in `LatentBriefingEngine` for future use |
| **Text Latent Briefing** | API-based models (Azure OpenAI, Claude) | **Current default** — API-based Azure OpenAI gpt-4o-mini. Implemented in `TextLatentBriefing` |

**Why Text Latent Briefing for now:** Our RLM orchestrator and workers use Azure OpenAI (API-only, no KV cache access). The text-level approach captures the core Latent Briefing insights (task-guided scoring, adaptive compression) using embeddings as an attention proxy. When/if we move to self-hosted workers (vLLM), the KV cache engine is ready.

#### Integration: `discover_fields_for_cluster_rlm_lb()`

A new field discovery function combines RLM + Latent Briefing:

```python
fields, briefing_meta = discover_fields_for_cluster_rlm_lb(
    cluster_label="Payment Terms",
    keywords=["payment", "net", "invoice"],
    chunk_texts=chunk_texts,
    config=config,
    briefing_tau=1.5,           # compaction aggressiveness
    briefing_max_tokens=2000,   # max tokens in briefing per worker call
    adaptive_tau=True,          # adjust tau based on iteration depth
)

# briefing_meta includes:
# - n_compactions: how many times trajectory was compacted
# - total_tokens_saved: cumulative token savings
# - avg_compression_ratio: mean compression across all compactions
# - per_compaction: [{original, compacted, ratio, tau, time_s}, ...]
```

---

### 4.4.3 Confidence Grounding for Field Extraction

> **[V2] Source-text grounding replaces LLM self-reported confidence**
>
> **Validated during hackathon.** Confidence scores are stable across scale (0.725 at 96-doc → 0.704 at 510-doc) because they are grounded in observable text features, not LLM self-assessment.

#### Problem Statement

LLM-reported confidence scores are unreliable — models exhibit systematic overconfidence and scores vary with prompt phrasing. A confidence system based on LLM self-report is not auditable.

#### Grounding Formula

```python
base_confidence = 0.5

# GROUNDING (strongest signals)
if value_verbatim_in_source_text:     conf += 0.30   # verbatim match
elif partial_word_match_in_source:     conf += 0.15   # 3+ words match

# VALUE QUALITY SIGNALS
if len(value) > 3:                     conf += 0.05   # non-trivial value
if contains_digits:                    conf += 0.05   # numeric = more reliable

# ERROR PENALTIES
if len(value) > 50:                    conf -= 0.10   # likely hallucinated passage
if value.startswith('{') or '[':       conf -= 0.15   # JSON fragment = extraction error

final_confidence = clamp(conf, 0.1, 1.0)
```

**Range:** [0.1, 1.0]. A perfectly grounded numeric value scores 0.90. An ungrounded long string scores 0.35.

#### Confidence Distribution (CUAD-510 Full Extraction)

| Confidence Band | Count | % | Interpretation |
|-----------------|-------|---|----------------|
| High (≥ 0.8) | 43,599 | 42.6% | Verbatim-grounded, reliable |
| Medium (0.6–0.8) | 24,075 | 23.5% | Partially grounded |
| Low (< 0.6) | 34,575 | 33.8% | Ungrounded — flag for review |
| **Total** | **102,249** | | **Avg: 0.704** |

**Stability across scale:**

| Dataset | Agreements | Avg Confidence | High (≥ 0.8) % |
|---------|-----------|---------------|-----------------|
| 96-doc sample | 96 | 0.725 | 44.1% |
| 200-doc sample | 200 | 0.712 | 43.3% |
| 510-doc full | 510 | 0.704 | 42.6% |

Confidence degrades only ~3% from 96→510 documents, confirming the grounding formula's stability.

#### Integration with Compliance Gating (§6.3)

The grounded confidence feeds directly into the confidence gating chain:

```
Confidence-grounded extraction → per-field confidence score
    → combined_confidence = cluster_conf × label_conf × extraction_conf
    → Compliance severity escalation (§6.3)
```

This replaces the LLM self-reported confidence originally specified in §6.3's extraction confidence step.

---

### 4.4.4 Field Discovery Evaluation Framework

#### Metrics Collected

| Metric | What It Measures | Source |
|--------|------------------|--------|
| Fields per cluster | Discovery breadth | Field count post-dedup |
| Jaccard overlap | Approach agreement | Intersection / Union of field name sets |
| Type agreement | Schema consistency | % of shared fields with matching data types |
| Fill rate | Extraction coverage | % of (doc × field) cells with non-null values |
| Avg confidence | Extraction reliability | Mean grounded confidence across all values |
| Latency per cluster | Cost/speed tradeoff | Wall-clock time for discovery |
| Failure rate | Reliability | % of clusters returning 0 fields |

#### Comparison Methodology

The `compare_field_discovery.py` script runs a controlled head-to-head:

1. Select N random clusters (configurable, tested at N=5 and N=20)
2. Run Standard discovery per cluster → record fields + timing
3. Run RLM discovery per cluster → record fields + timing  
4. (If `--include-hybrid`) Run 4-tier merge → record hybrid fields
5. For each approach, extract 5 sample documents per cluster
6. Compute Jaccard overlap, type agreement, fill rate, confidence
7. Output JSON + summary table for comparison dashboard

**Visualization:** Results are rendered in a 3-view D3.js dashboard (`comparison.html`):
- **Overview:** Radar chart (6 metrics), bar comparisons, scatter by latency
- **Per Cluster:** Side-by-side field lists with overlap highlighting
- **Field Analysis:** Type distribution, unique-to-each-approach breakdowns

---

### 4.5 Cluster-Conditioned Extraction

Instead of using the same generic prompt for all agreements, extraction prompts are **conditioned on the cluster's concept profile** (its label, keywords, representative examples, and known fields).

**Current approach (generic):**

```
Prompt: "Extract structured fields from this agreement section."
→ Model guesses what fields might exist → high hallucination risk, token waste
```

**Cluster-conditioned approach:**

```
Prompt: "This section belongs to a {cluster_label} clause (e.g., 'Payment Terms').
         Known fields for this clause type: {field_names_and_types}.
         Representative examples from similar agreements: {examples}.
         Extract values for each field. If a field is not present, return null."
→ Model knows exactly what to look for → higher precision, fewer tokens, lower hallucination
```

**Implementation:**
1. `SchematizeHandler` receives cluster metadata (label, fields, examples) from `OntologyService`
2. HyDE query generation uses the cluster label to produce more targeted hypothetical documents
3. Snippet retrieval runs dense (AIDB embedding search) and BM25 (keyword match on field names and known legal terms) in parallel, then merges results — ensuring both semantically similar and term-exact snippets are included
4. Extraction prompt is parameterized with the cluster's field schema
5. Over time, cluster-specific prompt templates can be refined based on extraction F1 per cluster

**Cost impact:** Cluster-conditioned extraction reduces average prompt length by ~40% (no need to enumerate all possible fields) and improves extraction F1 by constraining the output schema.

### 4.5.1 HyDE Query Strategy: Evolution from V1 to V2

**What HyDE does in V1:**

The current `SchematizeHandler` pipeline uses HyDE (Hypothetical Document Embedding) to bridge the embedding space gap between field definitions and actual contract text. Given a field name and description (meta-language like "Agreement start date"), HyDE generates a hypothetical document excerpt that reads like real contract text ("This Agreement shall be effective as of January 1, 2024"). This synthetic text is then embedded and used for ANN retrieval.

HyDE solves a real problem: raw field definitions and document text occupy different regions of embedding space. A field description *describes* what to find; HyDE produces text that *looks like* what to find. Dense embeddings reward surface-form similarity, so HyDE's output lands significantly closer to target chunks (~0.75–0.85 cosine similarity vs ~0.55–0.65 for raw definitions).

**Why V2 changes the calculus:**

V2 introduces several features that partially overlap with HyDE's purpose:

| V2 Feature | How It Helps Retrieval | Overlap with HyDE |
|------------|----------------------|-------------------|
| Cluster-conditioned extraction (§4.5) | Representative examples from the cluster are *actual contract text* — they naturally bridge the embedding gap | Representative examples do what HyDE does, but with real language instead of hallucinated language |
| BM25 sparse retrieval (§11.2.1) | Catches exact legal terms, statute references, defined terms that dense embeddings miss | Reduces HyDE's role as the sole retrieval quality booster |
| Macro→micro scoping (§3.1) | Retrieval is filtered to the relevant cluster's documents — smaller search space | Less need for a perfectly targeted query |
| Cluster concept profiles | Field names, types, value distributions, and keywords are known per cluster | More targeted input than V1's generic field description |

**The risk HyDE introduces:**

HyDE's hypothetical document can **mislead retrieval** if the LLM hallucinates incorrect legal phrasing. Example: for "Governing Law, " HyDE might generate "This Agreement shall be governed by the laws of the State of New York" — but the actual clause says "The parties submit to the exclusive jurisdiction of the courts of England and Wales." The HyDE embedding biases toward US-style language and may under-rank UK/international-style clauses. In V1 (broad corpus, no cluster scoping), this bias is diluted; in V2 (scoped to a specific macro cluster), it can meaningfully skew results.

**The cost:**

Each HyDE call is one LLM invocation per field per extraction run. With V2's expanded ontology (potentially hundreds of clause types × multiple fields each × periodic re-extraction), HyDE token costs scale multiplicatively. If cluster-representative-example retrieval achieves comparable recall, these are wasted tokens.

**Recommendation — Conditional HyDE:**

| Phase | HyDE Role | Rationale |
|-------|-----------|----------|
| Phase 1 | **Keep as-is** | Proven pipeline, no regression risk. V2 retrieval enhancements (BM25, cluster scoping) are additive |
| Phase 2 | **A/B test: HyDE vs cluster-example retrieval** | Use representative chunks from each cluster as retrieval queries directly. Measure recall@K and extraction F1 with and without HyDE |
| Phase 3+ | **Make HyDE conditional** | Invoke HyDE only when: (a) cluster has <5 representative examples (cold start), (b) field coverage_score < 30% (rare field, few examples), or (c) first extraction run for a new clause type. Skip HyDE when cluster is mature |
| Phase 5+ | **Replace with learned queries** | Fine-tuned query encoder (contrastive learning on field→snippet pairs from accumulated extraction data) replaces both HyDE and raw description retrieval |

**Implementation of conditional HyDE:**

```python
async def get_retrieval_query(field, cluster, aiflow_client):
    # Use representative examples if cluster is mature
    if len(cluster.representative_chunks) >= 5 and field.coverage_score > 0.3:
        return cluster.representative_chunks[0]  # already document-language
    # Fall back to HyDE for sparse/cold-start clusters
    return await generate_hyde_query(aiflow_client, field)
```

**Evaluation criteria for Phase 2 A/B test:**
* Primary: Extraction F1 per field (with vs without HyDE)
* Secondary: Recall@10 for snippet retrieval
* Cost: LLM tokens consumed per extraction run
* Statistical bar: HyDE must show >3% F1 improvement to justify the token cost; otherwise, deprecate for mature clusters

### 4.6 Agentic Ontology Refinement (Phase 5+)

Four specialized agents operate on the ontology:

**1. Discovery Agent**
* Input: New clusters or uncategorized chunks
* Output: Candidate clause types with proposed fields
* Prompt: "These paragraphs appear semantically related. Identify the clause type, describe its purpose, suggest fields to extract."

**2. Schema Generation Agent**
* Input: Discovered clause type + representative chunks
* Output: Structured field definitions with types and constraints
* Integrates with existing SchematizeHandler pipeline

**3. Critic / Validation Agent**
* Checks: coverage, field redundancy, field ambiguity, cluster coherence
* Example feedback: "Field 'payment_method' appears in only 20% of clauses — consider making it optional."

**4. Ontology Manager Agent**
* Maintains hierarchy: detects when clusters should merge, split, or form parent-child relationships
* Manages schema versioning

**Agent feedback loop:**

```
cluster → discovery agent → schema → extraction test → critic feedback → schema refinement
```

### 4.7 Hierarchical Topic Merging & Synonym Resolution (P0 — Phase 1–2)

> **[V2] Hierarchical Topic Merging & Synonym Resolution**
>
> **This is a P0 critical implementation item, not a spike.** Topic merging and synonym resolution directly address known V1 quality problems (duplicate clusters, fragmented ontologies) and must be implemented in Phase 1–2.

#### 4.7.1 Problem Statement

BERTopic frequently produces near-duplicate clusters that should be merged:

| Observed Duplicates | Should Merge To |
|--------------------|----------------|
| "Payment Terms" + "Net Payment Terms" + "Payment Conditions" | "Payment Terms" |
| "Liability Limitation" + "Limitation of Liability" + "Liability Cap" | "Liability Limitation" |
| "Confidentiality" + "Non-Disclosure" + "Confidential Information" | "Confidentiality" |
| "Termination for Cause" + "Termination for Breach" | "Termination for Cause" (or keep separate with hierarchy) |

Similarly, field names across clusters have synonyms that should resolve to canonical forms:

| Field Variants | Canonical |
|---------------|----------|
| `payment_days`, `net_terms`, `payment_period`, `days_to_pay` | `payment_days` |
| `effective_date`, `start_date`, `commencement_date` | `effective_date` |
| `governing_law`, `applicable_law`, `choice_of_law` | `governing_law` |

Without merging, the ontology becomes fragmented, extraction is duplicated, and users see confusing near-identical clusters.

#### 4.7.2 Hierarchical Topic Merging Architecture

**Two-phase approach:**

```
Phase A: Similarity-Based Merging
    Compute cluster centroid similarity matrix
    → Merge clusters above similarity threshold (default: 0.85)
    → Preserve provenance (merged_from, aliases)

Phase B: Hierarchical Organization
    For merged clusters with sub-structure:
    → Create parent-child relationships
    → "Termination" parent with children: "For Cause", "For Convenience", "Force Majeure"
```

**Merge algorithm:**

```python
class HierarchicalTopicMerger:
    """
    Merge semantically equivalent topics using single-linkage clustering.

    Design decisions:
    1. Use centroid cosine similarity (not keyword overlap) for robustness
    2. Single-linkage ensures transitive merging (A~B, B~C → merge A,B,C)
    3. Preserve all aliases for search and display
    4. Record merge provenance for audit trail
    """

    def __init__(
        self,
        merge_threshold: float = 0.85,
        min_cluster_size_for_hierarchy: int = 50
    ):
        self.merge_threshold = merge_threshold
        self.min_cluster_size_for_hierarchy = min_cluster_size_for_hierarchy

    def merge_topics(self, clusters: list[dict], centroids: np.ndarray) -> list[dict]:
        """Merge similar topics while preserving hierarchy potential."""
        # Compute pairwise centroid similarity
        sim_matrix = cosine_similarity(centroids)

        # Build merge groups via union-find
        merge_groups = self._union_find_merge(sim_matrix)

        # Create merged clusters
        merged = []
        for group in merge_groups:
            if len(group) == 1:
                merged.append(clusters[group[0]])
            else:
                merged.append(self._merge_cluster_group([clusters[i] for i in group]))

        # Check for hierarchical structure within large merged clusters
        for cluster in merged:
            if cluster.get("chunk_count", 0) >= self.min_cluster_size_for_hierarchy:
                cluster["hierarchy_candidate"] = True

        return merged

    def _merge_cluster_group(self, clusters: list[dict]) -> dict:
        """Merge multiple clusters into one canonical cluster."""
        # Pick primary: highest quality score or largest chunk count
        primary = max(clusters, key=lambda c: (
            c.get("quality_score", 0),
            c.get("chunk_count", 0)
        ))

        # Combine keywords (deduplicated, top 15)
        all_keywords = []
        for c in clusters:
            all_keywords.extend(c.get("keywords", []))
        combined_keywords = list(dict.fromkeys(all_keywords))[:15]

        # Collect aliases (other labels)
        aliases = [c["label"] for c in clusters if c["label"] != primary["label"]]

        return {
            **primary,
            "keywords": combined_keywords,
            "aliases": aliases,
            "merged_from": [c["cluster_id"] for c in clusters],
            "chunk_count": sum(c.get("chunk_count", 0) for c in clusters),
            "agreement_count": sum(c.get("agreement_count", 0) for c in clusters),
            "merge_confidence": self._compute_merge_confidence(clusters),
        }

    def _compute_merge_confidence(self, clusters: list[dict]) -> float:
        """Confidence that merged clusters are truly the same concept."""
        # Higher confidence when clusters have overlapping keywords
        keyword_sets = [set(c.get("keywords", [])) for c in clusters]
        if len(keyword_sets) < 2:
            return 1.0

        # Jaccard similarity of keywords
        intersection = set.intersection(*keyword_sets)
        union = set.union(*keyword_sets)
        return len(intersection) / len(union) if union else 0.0
```

#### 4.7.3 Synonym Resolution System

**Three-tier resolution strategy:**

| Tier | Method | Coverage | Precision |
|------|--------|----------|-----------|
| 1. Known synonyms | Curated lookup table | High for common fields | 100% |
| 2. Embedding similarity | Cluster field name embeddings | Medium | ~90% |
| 3. LLM confirmation | Ask LLM if two fields are synonymous | Low (expensive) | ~95% |

**Known synonym seed list (legal domain):**

```python
KNOWN_FIELD_SYNONYMS = {
    # Financial terms
    "payment_days": ["net_terms", "payment_period", "days_to_pay", "payment_terms_days", "net_days"],
    "liability_cap": ["limitation_of_liability", "max_liability", "liability_limit", "aggregate_liability"],
    "late_fee": ["late_payment_fee", "interest_rate", "penalty_rate", "late_charge"],

    # Date terms
    "effective_date": ["start_date", "commencement_date", "agreement_date", "execution_date"],
    "expiration_date": ["end_date", "termination_date", "term_end", "expiry_date"],
    "notice_period": ["notice_days", "termination_notice", "advance_notice", "notice_requirement"],

    # Party terms
    "contracting_parties": ["parties", "party_names", "counterparties", "signatories"],
    "governing_law": ["applicable_law", "choice_of_law", "jurisdiction", "governing_jurisdiction"],

    # Contract terms
    "auto_renewal": ["auto_renew", "automatic_renewal", "evergreen", "auto_extend"],
    "confidentiality_period": ["nda_term", "confidentiality_term", "secrecy_period", "confidential_period"],
    "indemnification": ["indemnity", "hold_harmless", "indemnify"],
}

class SynonymResolver:
    """Resolve field name variants to canonical forms."""

    def __init__(self, embed_model=None, similarity_threshold: float = 0.80):
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self._build_reverse_lookup()

    def _build_reverse_lookup(self):
        """Build variant → canonical mapping."""
        self.variant_to_canonical = {}
        for canonical, variants in KNOWN_FIELD_SYNONYMS.items():
            self.variant_to_canonical[canonical] = canonical
            for variant in variants:
                self.variant_to_canonical[variant] = canonical

    def resolve(self, field_name: str, field_definition: str = None) -> tuple[str, float]:
        """
        Resolve field name to canonical form.

        Returns: (canonical_name, confidence)
        """
        normalized = self._normalize(field_name)

        # Tier 1: Known synonyms (100% confidence)
        if normalized in self.variant_to_canonical:
            return self.variant_to_canonical[normalized], 1.0

        # Tier 2: Embedding similarity (if model available)
        if self.embed_model and field_definition:
            canonical, confidence = self._resolve_by_embedding(field_name, field_definition)
            if confidence >= self.similarity_threshold:
                return canonical, confidence

        # No match: return normalized original
        return normalized, 0.0

    def _normalize(self, name: str) -> str:
        """Normalize field name for lookup."""
        return name.lower().strip().replace(" ", "_").replace("-", "_")

    def _resolve_by_embedding(self, name: str, definition: str) -> tuple[str, float]:
        """Resolve via embedding similarity to known canonical fields."""
        query_text = f"{name}: {definition}"
        query_embedding = self.embed_model.encode([query_text], normalize_embeddings=True)

        # Compare against canonical field embeddings
        canonical_texts = [
            f"{canonical}: common legal contract field"
            for canonical in KNOWN_FIELD_SYNONYMS.keys()
        ]
        canonical_embeddings = self.embed_model.encode(canonical_texts, normalize_embeddings=True)

        similarities = cosine_similarity(query_embedding, canonical_embeddings)[0]
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score >= self.similarity_threshold:
            return list(KNOWN_FIELD_SYNONYMS.keys())[best_idx], float(best_score)

        return name, 0.0
```

#### 4.7.4 Hierarchical Ontology Structure

After merging, build explicit parent-child relationships:

```
Ontology (post-merge):
│
├── 📁 Termination (parent)
│   ├── 💼 Termination for Convenience
│   ├── 💼 Termination for Cause
│   └── 💼 Termination for Force Majeure
│
├── 📁 Payment Terms (merged)
│   ├── aliases: ["Net Payment Terms", "Payment Conditions"]
│   └── fields: [payment_days, payment_method, late_fee, currency]
│
├── 📁 Liability (parent)
│   ├── 💼 Liability Limitation (merged from "Limitation of Liability", "Liability Cap")
│   └── 💼 Indemnification (merged from "Indemnity", "Hold Harmless")
```

**Data model extension:**

```python
@dataclass
class ClauseType:
    clause_type_id: str
    label: str
    keywords: list[str]
    fields: list[Field]
    confidence: float

    # New fields for hierarchy and merging
    parent_clause_type_id: Optional[str] = None    # For hierarchy
    aliases: list[str] = field(default_factory=list)  # Merged labels
    merged_from: list[str] = field(default_factory=list)  # Source cluster IDs
    merge_confidence: float = 1.0                  # Confidence in merge correctness
    is_hierarchy_root: bool = False                # True for parent-only nodes
```

#### 4.7.5 Integration Points

| Component | Integration | Purpose |
|-----------|-------------|--------|
| `BERTopicClusterer` | Post-processing hook | Merge similar topics before labeling |
| `OntologyService` | Store merged structure | Persist aliases, hierarchy, merge provenance |
| `SchematizeHandler` | Field synonym resolution | Map extracted fields to canonical names |
| Ontology UI | Display merged clusters | Show aliases, hierarchy tree, merge confidence |
| Search API | Alias expansion | Query "Net Payment Terms" finds "Payment Terms" |

#### 4.7.6 Implementation Timeline & Success Criteria

| Phase | Milestone | Success Criteria |
|-------|-----------|------------------|
| Phase 1 (Hackathon) | Prototype `HierarchicalTopicMerger` + `SynonymResolver` | Working demo showing merged clusters |
| Phase 1 (Week 3-4) | Production implementation | Integrated into clustering pipeline |
| Phase 2 (Week 5-8) | Synonym table expansion | 100+ canonical field mappings |
| Phase 2 (Week 9-12) | UI integration | Merge visualization in ontology viewer |

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Cluster reduction rate** | 15–30% | (clusters_before - clusters_after) / clusters_before |
| **Merge precision** | ≥90% | Manual review: % of merges that are correct |
| **Merge recall** | ≥80% | Manual review: % of true duplicates merged |
| **Field synonym coverage** | ≥70% | % of field variants resolved to canonical |
| **Ontology coherence** | Improved | User rating of merged ontology vs original |

**Implementation is complete when:** Merge precision ≥90% AND cluster reduction ≥15%.

#### 4.7.7 Configuration & Tuning

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `merge_threshold` | 0.85 | 0.75–0.95 | Higher = fewer merges, higher precision |
| `synonym_threshold` | 0.80 | 0.70–0.90 | Higher = fewer synonym matches |
| `min_cluster_for_hierarchy` | 50 | 20–100 | Min chunks to consider hierarchy |
| `max_aliases` | 10 | 5–20 | Cap aliases per merged cluster |

---

## 5. Intent Layer (Future Differentiator — Phase 5+)

> **Framing note:** Intent is the long-term north star for this platform. It is NOT a near-term dependency. Phases 1–4 deliver full value (clause intelligence, compliance, recommendations, reasoning) without the intent layer. Intent is designed here to inform the architecture, not to block the first release.
>
> **Validation requirement before building:** Intent must demonstrate measurable improvement over the simpler recommendation baseline (cluster frequency + org preference + policy score). If A/B testing shows no statistically significant lift from intent-aligned scoring, this layer is deprioritized.

### 5.1 Concept

Intent represents **why** a clause exists and **what tradeoff** it encodes. This moves the system from structural understanding to decision intelligence.

| Layer | Question Answered |
|-------|-------------------|
| Embedding | "What does this text mean?" |
| Cluster | "What group does it belong to?" |
| Ontology | "What clause type is it?" |
| **Intent** | **"What is it trying to achieve?"** |

### 5.2 Intent Schema

```
IntentDefinition {
    clause_type_id: string
    intent_label: string           // "cash_flow_optimization"
    description: string
    dimensions: list[IntentDimension]
}

IntentDimension {
    name: string                   // "cash_flow"
    description: string
    polarity: enum[higher_better, lower_better, neutral]
}

FieldIntentMapping {
    field_id: string
    dimension_effects: dict        // {"30": {"cash_flow": 0.2, "vendor_friction": 0.1},
                                   //  "60": {"cash_flow": 0.8, "vendor_friction": 0.6}}
}
```

**Example:**

```
ClauseType: Payment Terms
Intent: balance_cash_flow_and_vendor_relationship
Dimensions:
  - cash_flow (higher_better)
  - vendor_friction (lower_better)
  - risk (lower_better)

Field Mappings:
  payment_days:
    30 → {cash_flow: 0.2, vendor_friction: 0.1}    // vendor-friendly
    45 → {cash_flow: 0.5, vendor_friction: 0.3}    // balanced
    60 → {cash_flow: 0.8, vendor_friction: 0.6}    // cash-optimized
```

### 5.3 Intent Construction

**Phase 1: Rule-based (MVP)**

Define 2–3 intent dimensions per clause type manually for the top 5–10 clause types. Map field values to dimension scores using simple rules.

**Phase 2: Agent-driven**

* Intent Discovery Agent: analyzes cluster keywords + representative chunks → proposes intent labels + dimensions
* Mapping Agent: maps fields → dimension effects based on corpus statistics
* Evolution Agent: updates intent definitions as new data arrives

**Phase 3: Learned intents**

Derive intent vectors from user behavior:
* Accepted clause → positive intent signal
* Rejected clause → negative intent signal
* Edited clause → delta intent signal

### 5.4 Document Intent Profile

```
document_intent = Σ (cluster_presence × intent_vector)

Example:
  Agreement #A123:
    cash_flow: 0.6 (moderately optimized)
    risk_tolerance: 0.3 (conservative)
    vendor_friendliness: 0.7 (vendor-friendly overall)
```

### 5.5 Intent-Aware Recommendations

Instead of recommending by similarity alone:

```
Traditional: "this looks like other contracts"
Intent-aware: "this achieves your goal"

Example:
  User goal: "reduce financial risk"
  System:
    → increase liability cap constraints
    → shorten payment terms
    → add termination flexibility
```

### 5.6 Why Embeddings Are Not Required for Intent

Intent is a **structured abstraction**. Embeddings are a tool to *reach* that abstraction, not to *represent* it.

| Purpose | Needs Embeddings? |
|---------|-------------------|
| Connecting raw text → intent | Yes (via clustering) |
| Representing intent itself | No (symbolic/structured) |
| Generalizing to new phrasing | Yes (embeddings + clustering) |
| Personalizing intent | No (preference vectors) |

Once clusters are mapped to intents, the intent layer operates on structured data (statistics, rules, preferences) — not embeddings.

---

## 6. Policy & Compliance Engine

### 6.1 Policy Model

```
Policy {
    policy_id: string
    organization_id: string
    rules: list[PolicyRule]
    created_at: timestamp
    version: int
}

PolicyRule {
    clause_type_id: string          // "payment_terms"
    field_id: string                // "payment_days"
    operator: enum[<=, >=, ==, !=, in, not_in, exists, not_exists]
    threshold: any                  // 45
    severity: enum[warning, violation, block]
    message: string
}
```

**Example:**

```json
{
    "clause_type_id": "liability",
    "field_id": "liability_cap",
    "operator": "!=",
    "threshold": "unlimited",
    "severity": "violation",
    "message": "Company policy requires capped liability"
}
```

### 6.2 Compliance Check Pipeline

```
Agreement uploaded
    ↓
Clause detection (via cluster assignment)
    ↓
Confidence gating (see below)
    ↓
Field extraction (via SchematizeHandler)
    ↓
Policy rule evaluation
    ↓
Violation report + suggested alternatives (human review required)
```

### 6.3 Confidence Gating (Error Propagation Control)

The cluster → clause → field → policy chain is the **primary technical risk** in the system. Each step has noise, and errors compound:

```
Noisy cluster → wrong clause type → wrong field extraction → false compliance result
```

To prevent this, every step emits a confidence score, and the chain is gated:

| Step | Confidence Source | Gate Threshold | Below-Threshold Behavior |
|------|-------------------|----------------|-------------------------|
| Cluster assignment | Cosine similarity to cluster centroid | ≥ 0.65 | Flagged as "unclassified" — excluded from compliance, available for discovery |
| Clause-type label | LLM labeling confidence + cluster coherence score | ≥ 0.75 | Clause type shown as "provisional" — compliance rules not applied |
| Field extraction | LLM extraction confidence (from SchematizeHandler) | ≥ 0.70 | Field shown as "low confidence" — compliance check emits warning, not violation |
| Policy evaluation | Product of upstream confidences | ≥ 0.50 combined | Finding downgraded from "violation" to "review needed" |

**Compliance severity escalation:**

```
combined_confidence = cluster_conf × label_conf × extraction_conf

if combined_confidence ≥ 0.50 AND policy_rule.severity == "violation":
    emit finding as: "⚠️ Violation (confidence: {combined_confidence:.0%})"
elif combined_confidence ≥ 0.35:
    emit finding as: "🟡 Review needed (confidence: {combined_confidence:.0%})"
else:
    suppress finding (log for audit)
```

All thresholds are configurable per organization. Conservative organizations can raise thresholds; exploratory pilots can lower them.

### 6.4 Risk Scoring

```
risk_score(clause) =
    w1 * policy_violation_score
  + w2 * rarity_score            // how unusual vs cluster distribution
  + w3 * deviation_from_org      // how far from organization norm
```

---

## 7. Recommendation Engine

### 7.1 Scoring Model

```
recommendation_score(clause) =
    α * cluster_frequency_score   // how common in similar contracts
  + β * org_preference_score      // how aligned with org history
  + γ * policy_compliance_score   // does it satisfy policies
  + δ * intent_alignment_score    // does it match user's goals (Phase 5+)
```

### 7.2 Sources

| Signal | Source | Available |
|--------|--------|-----------|
| Cluster frequency | BERTopic topic stats | Phase 1 |
| Field value distribution | Schema extraction pipeline | Phase 2 |
| Organization preference | Tracked edits/accepts | Phase 4 |
| Intent alignment | Intent layer | Phase 5 |
| Industry benchmark | Cross-customer aggregation | Phase 5 |

### 7.3 Similar Contract Retrieval

```
query_clause
  ↓
├─ Dense: MiniLM embedding → ANN search within cluster → Top-K
├─ Sparse: BM25 keyword search within cluster → Top-K
  ↓
RRF merge → Final Top-K similar clauses
```

Cluster-aware filtering reduces search space and improves relevance. The BM25 path ensures that queries containing specific defined terms, statute references, or threshold values (e.g., "$1M aggregate liability") return exact matches that dense retrieval alone would rank lower.

---

## 8. Reasoning & Explainability Layer

### 8.1 Reasoning Trace Schema

```
RecommendationTrace {
    recommendation_id: string
    clause_type_id: string
    evidence: list[EvidenceNode]
    scores: dict                    // {cluster_support: 0.72, org_preference: 0.81, ...}
    timestamp: timestamp
}

EvidenceNode {
    type: enum[cluster_stat, org_preference, policy, similar_contract, industry_benchmark]
    description: string
    value: any
    confidence: float
}
```

### 8.2 Explanation Generation

**Do NOT let the LLM invent explanations.**

Instead:
1. System produces structured reasoning traces from stored data
2. LLM converts traces into natural language (templated)

```
Trace:
  {cluster_support: 2300, org_preference: 0.6, policy_score: 1.0}

Generated explanation:
  "Recommended because:
    • Supported by 2,300 similar clauses in cluster
    • Aligns with 60% of your organization's contracts
    • Fully compliant with company policy"
```

### 8.3 Counterfactual Explanations

```
"If liability remains unlimited:
  • 92% of peers would reject this
  • Increases negotiation time by ~30%
  • Violates company policy (severity: violation)"
```

---

## 9. Personalization Engine

### 9.1 Preference Data Model

**Design principle:** Legal preferences are **contextual**, not global. A user's preferred payment terms vary by contract type, counterparty, jurisdiction, business unit, and deal size. The preference model must capture this or risk overfitting to irrelevant signals.

```
OrganizationPreference {
    org_id: string
    clause_preferences: dict[clause_type_id → ContextualFieldPreference]
    updated_at: timestamp
}

ContextualFieldPreference {
    field_id: string
    global_distribution: dict      // {"45": 0.6, "30": 0.2, "60": 0.2}
    context_overrides: list[ContextOverride]
    trend: enum[increasing, stable, decreasing]
}

ContextOverride {
    context_key: string             // "contract_type", "jurisdiction", "counterparty_industry"
    context_value: string           // "SaaS", "EU", "financial_services"
    distribution: dict              // {"30": 0.7, "45": 0.3}
    sample_size: int                // number of agreements backing this override
}

UserPreference {
    user_id: string
    org_id: string
    clause_overrides: dict          // where user differs from org
    feedback_history: list[FeedbackEvent]
}

FeedbackEvent {
    clause_type_id: string
    action: enum[accept, reject, edit]
    old_value: any
    new_value: any
    context: dict                   // {"contract_type": "SaaS", "jurisdiction": "US"}
    timestamp: timestamp
}
```

### 9.2 Contextual Preference Learning

**Organization level (with context):**

```
org_preference[payment_days | contract_type=SaaS] = {30: 0.7, 45: 0.3}
org_preference[payment_days | contract_type=Enterprise] = {45: 0.5, 60: 0.5}
org_preference[payment_days | global] = {30: 0.2, 45: 0.6, 60: 0.2}
```

Preference lookup priority:
1. Context-specific override (if sample_size ≥ 10)
2. Global org distribution
3. Industry fallback (cold start)

**User level:**

```
user edits payment_days: 30 → 45 (3 times, all in SaaS contracts)
    → user_preference[payment_days | SaaS] weighted toward 45
    → user_preference[payment_days | global] NOT updated (context-specific)
```

### 9.3 Cold Start

For new users/organizations:
* Fall back to industry-level distributions (from cross-cluster stats)
* Fall back to similar organization profiles (by industry, size, jurisdiction)

---

## 10. Feature Store

### 10.1 Purpose

Persist computed signals for fast retrieval. Avoid recomputing everything on each request.

### 10.2 Stored Features

```
TopicStats {
    topic_id: int
    clause_type_id: string
    cluster_size: int
    field_distributions: dict[field_id → value_distribution]
    last_updated: timestamp
}

OrgClauseStats {
    org_id: string
    clause_type_id: string
    frequency: float
    field_distributions: dict
    violation_count: int
}

ReasoningCache {
    recommendation_id: string
    trace: RecommendationTrace
    ttl: timestamp
}
```

### 10.3 Update Pipeline

```
User feedback event
    ↓
Event stream (Service Bus)
    ↓
Feature updater (async worker)
    ↓
Feature store update
    ↓
Recommendation engine uses updated features
```

---

## 11. Hybrid Retrieval Layer (Phase 5+)

### 11.1 Architecture

```
Query
  ↓
┌────────────────────────────────────────────┐
│  Stage 1 — Parallel Candidate Retrieval    │
│  ├─ Dense: embedding model → ANN → Top-100 │
│  └─ Sparse: BM25 (inverted index) → Top-100│
└────────────────────────────────────────────┘
  ↓
Reciprocal Rank Fusion (RRF) → Top-100 merged candidates
  ↓
Cluster Filter (restrict to relevant clusters) → Top-50
  ↓
Cross-encoder Reranking (query+doc joint scoring) → Top-5
  ↓
Intent-aligned scoring
  ↓
Final recommendation
```

### 11.2 Dense Retrieval

```python
query_vec = minilm_model.encode(query)
dense_candidates = faiss_index.search(query_vec, top_k=100)
```

### 11.2.1 Sparse Retrieval (BM25)

```python
bm25_candidates = bm25_index.search(query_text, top_k=100)
```

**Why BM25 is critical for legal retrieval:**

Dense embeddings capture semantic similarity but can miss exact-match signals that are decisive in legal contexts:

| Signal Type | Example | Dense Embedding | BM25 |
|-------------|---------|-----------------|------|
| Statute references | "Section 2-207 UCC" | Approximate match | Exact match |
| Defined terms | "Capital Expenditure" (as defined in §1.1) | Semantic similarity to spending | Exact term match |
| Named entities | "LIBOR", "SOFR", "Delaware" | Blurry embedding | Precise hit |
| Boilerplate identifiers | "AS-IS", "WITHOUT WARRANTY" | Averaged away | Direct match |
| Numeric thresholds | "$5, 000, 000 aggregate" | Lost in embedding | Token-level match |

**BM25 index construction:**
* Built from the same chunked corpus used for embeddings
* Tokenized with legal-aware preprocessing (preserve §, $, defined-term casing)
* Stored as an inverted index (Elasticsearch, Lucene, or lightweight libs like `rank_bm25` / `bm25s`)
* Updated incrementally as new agreements are ingested

### 11.2.2 Reciprocal Rank Fusion (RRF)

Merge dense and BM25 candidate lists without tuning weights:

```python
def rrf_score(doc, rankings, k=60):
    return sum(1.0 / (k + rank) for rank in rankings[doc])

merged = sorted(all_candidates, key=lambda d: rrf_score(d, {"dense": dense_ranks, "bm25": bm25_ranks}), reverse=True)[:100]
```

RRF is preferred over linear combination because it is rank-based (no score normalization needed) and robust to score distribution differences between dense and sparse retrievers.

### 11.3 Cross-Encoder Reranking

Instead of ColBERT (multi-vector, token-level matching), use a **cross-encoder reranker** for precision scoring. A cross-encoder concatenates the query and document text, feeds them jointly through a transformer, and outputs a single relevance score:

```
Input:  [CLS] query text [SEP] candidate document text [CLS]
  ↓
Transformer (full cross-attention between query and document)
  ↓
Single relevance score (0–1)
```

**Why cross-encoder over ColBERT:**

| Property | ColBERT (multi-vector) | Cross-Encoder Reranker |
|----------|----------------------|------------------------|
| Storage | 200 tokens × 128 dims × 4 bytes = **100 KB per chunk** (67x dense) | **Zero** — no precomputed embeddings |
| At 10M chunks | ~1 TB stored token vectors | 0 bytes |
| Accuracy | Very good | **Best** — full cross-attention sees both texts jointly |
| Latency (50 candidates) | ~10ms (precomputed dot products) | ~50–100ms (50 forward passes) |
| Legal nuance | Good token-level matching | **Better** — understands "aggregate liability is limited to" matches "liability shall not exceed" even with no token overlap |
| Candidate model | `cross-encoder/ms-marco-MiniLM-L-6-v2` (general), or fine-tuned on legal pairs | |

At your retrieval volume (top-50 candidates after cluster filtering), cross-encoder latency (~50–100ms) is well within acceptable bounds. ColBERT's latency advantage only matters when reranking thousands of candidates.

**Why NOT use MiniLM token vectors for MaxSim ("poor man's ColBERT"):**

MiniLM's per-token vectors are a byproduct of training — they were optimized to be *mean-pooled* into a single sentence vector, not to function as independent retrieval vectors. Using them with MaxSim scoring gives **3x the storage cost of ColBERT** (384 dims vs 128 dims per token) **for worse retrieval quality**. ColBERT's entire training procedure (distillation + MaxSim loss) is designed to make each token vector individually useful. MiniLM's are not.

| Approach | Quality | Storage per chunk | Verdict |
|----------|---------|-------------------|---------|
| Cross-encoder | Best | 0 bytes | **Use this** |
| ColBERT | Very good | ~100 KB | Overkill for top-50 reranking |
| MiniLM-MaxSim | Poor | ~307 KB | Worst of all worlds — don't do this |

### 11.4 Hybrid Score Combination

```
final_score =
    α * dense_similarity_score
  + β * bm25_score
  + γ * cross_encoder_rerank_score
  + δ * cluster_membership_score
  + ε * intent_alignment_score
```

Use RRF (Section 11.2.2) for the dense + BM25 fusion stage. Reserve the weighted combination for post-reranking scoring where cross-encoder and intent scores are on comparable scales.

### 11.5 Cost Management

* **BM25 index:** Lightweight inverted index; negligible memory vs vector index. Incremental updates are O(doc_length), no re-embedding cost
* **Cross-encoder reranker:** Zero storage overhead. Compute cost is ~50–100ms per 50 candidates — runs only after cluster filtering narrows the candidate set
* Dense + BM25 run in parallel at Stage 1, so latency = max(dense, BM25) ≈ dense alone (BM25 is typically faster)
* Cluster-aware filtering reduces candidate set before reranking
* **No per-token embeddings stored** (unlike ColBERT, which would require ~100 KB per chunk × millions of chunks)

---

## 12. Continuous Learning Pipeline

### 12.1 Feedback Signals

| Signal | Source | What It Updates |
|--------|--------|----------------|
| Recommendation accepted | User action | org_preferences, feature_store |
| Recommendation rejected | User action | negative preference signal |
| Field value edited | User action | field_distribution, user_preference |
| Field definition updated | User action | field schema (name, type, description, examples, normalization) |
| Derived field created | User action | field schema (computation rule, source fields) |
| Clause added/removed | User action | clause frequency stats |
| Cluster used in compliance | System | policy effectiveness metrics |
| Snippet retrieval result | System | contrastive training pairs (see §12.2) |

### 12.2 Contrastive Learning Data Collection (Phase 2–3)

**Purpose:** Accumulate training data for a domain-adapted embedding model that understands legal-specific semantics.

**Natural training pairs from system operation:**

Every extraction run produces implicit supervision:

| Pair Type | Source | Example |
|-----------|--------|---------|
| **(field description, source snippet)** — positive | Extraction pipeline: snippet that *contained* the extracted value | ("liability cap", "In no event shall aggregate liability exceed 2x fees paid") |
| **(field description, random snippet from same doc)** — hard negative | Same document, wrong section | ("liability cap", "This Agreement shall be governed by Delaware law") |
| **(clause A, clause A')** from same cluster — positive | Two chunks assigned to the same micro cluster | Both are "Payment Terms" clauses from different agreements |
| **(clause A, clause B)** from different clusters — negative | Chunks from different micro clusters | "Payment Terms" chunk vs "Termination" chunk |
| **(accepted snippet, rejected snippet)** — preference | User edits or rejects an extraction | User corrects a value → the correct snippet is positive, the originally retrieved one is hard negative |

**Collection pipeline:**

```python
# Stored automatically during extraction runs
TrainingTriple {
    query: string              # field description or HyDE query
    positive_snippet: string   # snippet that contained the correct extracted value
    negative_snippet: string   # snippet from same doc that did NOT contain the value
    cluster_id: string
    confidence: float          # extraction confidence for weighting
    source: enum[extraction, user_edit, cluster_assignment]
}
```

**Volume requirements:**
* ~10K–50K triples needed to outperform a general pretrained model
* At ~5 fields/clause type × ~50 clause types × ~100 docs/cluster = ~25K triples per full extraction run
* One or two full-corpus extraction runs should produce enough data

**Fine-tuning approach:**
* Base model: Nomic v1.5 (or MiniLM if Nomic eval doesn't pass)
* Loss: MultipleNegativesRankingLoss (sentence-transformers) with in-batch negatives + mined hard negatives
* Validation: Hold out 20% of triples, measure recall@10 improvement vs base model
* Deployment: Replace base embedding model in AIDB; re-embed corpus with fine-tuned model

**When to fine-tune:**
* Not before Phase 3 — need sufficient extraction data and a baseline to beat
* Trigger: accumulated >10K triples AND retrieval recall@10 on legal eval set is <0.85 with base model

### 12.3 Field Definition Lifecycle (CRUD)

Users need more than accept/reject. They need full control over auto-discovered field definitions.

**Supported operations:**

| Operation | Description | Example |
|-----------|-------------|----------|
| **Rename** | Change field label | `payment_days` → `net_payment_terms` |
| **Retype** | Change field return type | `string` → `int` |
| **Redefine** | Update description + extraction guidance | "Number of days until payment is due from invoice date" |
| **Add examples** | Provide gold examples for extraction | `{"text": "net 30 days", "value": 30}` |
| **Set normalization** | Define how raw values map to canonical form | `"net thirty" → 30` , `"net 45 days" → 45` |
| **Create derived field** | Define computed/composite field from base fields | `total_contract_value = unit_price × quantity × term_years` |
| **Delete / hide** | Remove field from cluster schema (soft delete) | Field hidden from UI but retained in version history |
| **Scope** | Apply field to single cluster or across all clusters | `governing_law` applies globally; `auto_renew` applies to renewal cluster only |

**Derived fields** are particularly important for legal intelligence:

```
Clause: Auto-Renewal
  Base fields:
    - renewal_term: "12 months"
    - notice_period: "60 days"
    - auto_renew: true
  Derived:
    - renewal_type: enum(auto, manual) ← computed from auto_renew
    - last_opt_out_date: date ← computed from expiration - notice_period
```

**Storage:** Field definitions stored in AIDB via `OntologyService` . Each update creates a new `SchemaVersion` (see Section 14). User-modified field definitions are flagged as `source: user_override` vs `source: auto_discovered` .

**Feedback quality evaluation:** When a user modifies a field definition and re-runs extraction on the sample set, the system computes ROUGE/BLEU between the original and updated extraction results to quantify improvement. This score is logged for ML team review.

### 12.4 Composite Field Engine (Phase 3+)

#### 12.4.1 Motivation: Learning Efficiency Through Composition

The core insight: **if a customer defines a composite field as a formula over base fields, the composite inherits the accumulated learning of every base field it references — for free.**

Consider a customer who has spent months refining `unit_price` extraction (200 corrections, normalization rules like "per seat" → numeric, gold examples) and `term_years` extraction (similar refinement). If they now want TCV:

| Approach | How TCV Is Obtained | Inherits Base Field Learning? | LLM Cost | Accuracy |
|----------|--------------------|-----------------------------|----------|----------|
| **Direct LLM extraction** | New HyDE query, new LLM call, new learning loop from scratch | No — independent pipeline | 1 LLM call per agreement | LLM must infer math; unreliable |
| **Composite by formula** | `TCV = unit_price × term_years` , computed from already-extracted values | **Yes — every correction to unit_price or term_years automatically improves TCV** | Zero | Deterministic arithmetic — correct if inputs are correct |

Without composition, redefining TCV as an independent extraction throws away months of accumulated intelligence on its component fields. With composition, a single correction to `unit_price` compounds upward through every composite that references it.

#### 12.4.2 Data Model

```
CompositeFieldDefinition {
    field_id: string
    name: string                          // "total_contract_value"
    expression: string                    // "unit_price * quantity * term_years"
    source_field_refs: list[FieldRef]     // [ref(unit_price), ref(quantity), ref(term_years)]
    result_type: enum[int, float, date, bool, string]
    clause_type_id: string?              // scoped to a cluster, or null for global
    created_by: string                   // user_id
    created_at: timestamp
    version: int
}

FieldRef {
    field_id: string                     // pointer to base or composite field
    clause_type_id: string              // which cluster's extraction to use
    required: bool                      // if true, composite is null when base is null
}
```

**Key design decision: references by field_id, not by value.** The formula stores pointers to base fields. At evaluation time, it reads the *current* extracted value of each base field. When a base field's value is corrected, all composites referencing it automatically reflect the correction — no re-extraction, no re-computation trigger needed (lazy evaluation on read).

#### 12.4.3 Expression Language

Support a **restricted arithmetic DSL** — not a general-purpose language:

| Category | Operators | Example |
|----------|-----------|---------|
| Arithmetic | `+` , `-` , `*` , `/` | `unit_price * quantity` |
| Date math | `date_add` , `date_sub` , `date_diff` | `date_sub(expiration_date, notice_period)` |
| Conditionals | `if(cond, then, else)` | `if(auto_renew, renewal_term, 0)` |
| Aggregation | `sum` , `min` , `max` , `avg` , `count` | `sum(line_item_values)` (over array fields) |
| Coercion | `to_int` , `to_float` , `to_date` | `to_int(payment_days_text)` |
| Null handling | `coalesce(a, b)` , `is_null(a)` | `coalesce(stated_tcv, unit_price * term)` |

**Not supported (by design):** string concatenation, regex, arbitrary function calls, loops, external API calls. This is a formula engine, not a scripting language. Keeps it safe, auditable, and side-effect-free.

#### 12.4.4 Dependency DAG & Evaluation

Composites can reference other composites, forming a directed acyclic graph (DAG):

```
unit_price (base, LLM-extracted)
quantity (base, LLM-extracted)
term_years (base, LLM-extracted)
    ↓
annual_cost = unit_price × quantity          (level-1 composite)
    ↓
TCV = annual_cost × term_years              (level-2 composite)
cost_per_month = annual_cost / 12           (level-1 composite)
```

**Evaluation rules:**
1. **Lazy evaluation:** Composites are computed on read, not eagerly on every base field change. No background recomputation jobs.
2. **Cycle detection:** At definition time, validate the DAG has no cycles. Reject any formula that creates a circular dependency.
3. **Max depth:** Cap at 5 levels to prevent deep chains that are hard to debug. (Revisit if real use cases exceed this.)
4. **Null propagation:** If any `required: true` source field is null, the composite evaluates to null. Optional source fields use `coalesce` semantics.

#### 12.4.5 Feedback Propagation

This is the core value proposition:

```
User corrects unit_price: $50,000 → $55,000
    ↓
annual_cost (= unit_price × quantity) automatically reflects $55K × 100 = $5.5M
    ↓
TCV (= annual_cost × term_years) automatically reflects $5.5M × 3 = $16.5M
    ↓
cost_per_month (= annual_cost / 12) automatically reflects $458,333

One correction → three composites updated → zero additional LLM calls
```

**What propagates:**
* Value corrections (user fixes a base field value)
* Normalization improvements (user adds a rule like "net thirty" → 30)
* Confidence improvements (more examples → better extraction → better inputs to composites)
* Schema changes (user retypes a base field from string to int → composites get cleaner inputs)

**What does NOT propagate (and shouldn't):**
* Re-extraction triggers — composites don't cause base fields to re-extract
* Formula changes — changing a formula is a schema change, versioned separately

#### 12.4.6 Cross-Validation with Direct Extraction

When a composite field's value also appears literally in the agreement text (e.g., "Total Contract Value: $16, 500, 000"), offer optional cross-validation:

```
computed_TCV = unit_price × quantity × term_years = $16,500,000
extracted_TCV = LLM extracts "Total Contract Value" from text = $16,500,000
    → Match: high confidence ✓

computed_TCV = $16,500,000
extracted_TCV = $15,000,000
    → Mismatch: flag for review — either a base field is wrong, or the document has an error
```

This is opt-in per composite field. When enabled, it provides a **built-in integrity check** that neither approach alone offers.

#### 12.4.7 Storage & Concept Requirements

**New concepts required:**

| Concept | New? | Storage Impact |
|---------|------|----------------|
| `CompositeFieldDefinition` table | Yes — new table in AIDB | ~500 bytes per definition. At 100 composites per org × 10K orgs = ~500 MB max. Negligible. |
| `FieldRef` (dependency edges) | Yes — new table for DAG edges | ~100 bytes per edge. At ~3 refs per composite × 100 composites × 10K orgs = ~30 MB. Negligible. |
| Computed values cache | **No** — composites are lazy-evaluated on read, not stored | 0 bytes. Computed at query time from already-stored base field values. |
| Expression parser | Yes — new code, not storage | Lightweight DSL parser (AST). No DB impact. |
| DAG validator | Yes — new code, not storage | Cycle detection at definition time. No DB impact. |
| Schema versioning for composites | No — uses existing `SchemaVersion` (§14) | Composite definitions versioned alongside base field schema. Marginal increase to existing version table. |

**Total new storage: <1 GB across all accounts.** Composite definitions are metadata (formulas + refs). The actual *values* are never stored — they're computed on the fly from base field extractions that already exist in AIDB.

**No new embedding storage.** Composites don't go through the embedding/retrieval pipeline at all. They don't need HyDE, snippet retrieval, or ANN search. They're pure arithmetic on already-extracted values.

**New infrastructure:**
* Expression parser (restricted DSL → AST → evaluator). ~500 LOC. No external dependencies.
* DAG validator (topological sort + cycle detection). ~100 LOC.
* `OntologyService` extension: CRUD for `CompositeFieldDefinition`. Follows existing gRPC pattern.

### 12.5 Update Loop

```
User interaction
    ↓
Event published (Service Bus)
    ↓
Feature store updated (async)
    ↓
Recommendation weights refreshed
    ↓
Next recommendation is better
```

### 12.6 Periodic Batch Updates

* **Monthly:** Re-compute org-level preference distributions
* **Quarterly:** Re-evaluate cluster quality; trigger re-clustering if drift detected
* **On-demand:** Ontology review when new clause types emerge

### 12.7 Contrastive Learning Loop (Phase 6+)

```
Accepted clauses → positive pairs
Rejected clauses → negative pairs
Hard negatives: similar but different clause types

Train custom embedding model:
    maximize similarity(accepted, user_profile)
    minimize similarity(rejected, user_profile)

Result: domain-specific embeddings that improve clustering quality
```

---

## 13. Graph-Based Extensions & Advanced Discovery (Phase 6+)

> **Framing note:** This section describes optional expansions that become relevant only if signal quality (cluster purity, extraction F1, recommendation accept rate) justifies the infrastructure investment. In a VP/CTO review, treat this as "future expansion if data quality warrants it" — not a near-term commitment.

### 13.1 When to Introduce Graphs

**Do NOT introduce graph infrastructure until:**
* Multi-hop reasoning is required (Clause A depends on Clause B)
* Cross-clause intelligence is needed (changing payment terms affects late fee clauses)
* Ontology becomes hierarchical (Termination → {convenience, breach, force majeure})

**Until then:** Simple relational tables + cluster metadata are sufficient.

### 13.2 Lightweight Graph (via SQL)

Before introducing Neo4j or a full graph DB, represent relationships in SQL:

```sql
CREATE TABLE clause_relationships (
    source_clause_type_id TEXT,
    target_clause_type_id TEXT,
    relationship_type TEXT,        -- 'depends_on', 'conflicts_with', 'related_to'
    strength FLOAT,
    evidence_count INT
);
```

### 13.3 Full Graph, GNNs, Community Detection, and Self-Evolving Discovery

If the system reaches Phase 6 maturity, the following extensions are available:

* **Multi-layer graph architecture:** Paragraph → Topic → Clause → Field → User/Org → Intent nodes with typed edges (SIMILAR_TO, BELONGS_TO_TOPIC, HAS_FIELD, SUBTYPE_OF, COMPLIES_WITH, PREFERS, INFLUENCES)
* **Graph Neural Networks (GNNs):** Learn cross-entity patterns for cold-start personalization (e.g., fintech companies → prefer arbitration → require data security clauses)
* **Community detection (Louvain/Leiden):** Alternative to BERTopic for production ontology evolution where incremental updates and agentic pipeline integration are priorities
* **Self-evolving clause discovery:** Anomaly detection on new documents → micro-cluster formation → agent-evaluated candidate clause types → critic validation → ontology update
* **Self-play evaluation:** Generator agent proposes schemas; critic agent challenges overlap, extractability, and coherence; refinement agent iterates

These are documented in detail to inform architecture decisions, but none are dependencies for Phases 1–5.

---

## 14. Schema Versioning

### 14.1 Why It Matters

Legal schemas evolve as regulations change, new clause types emerge, and field definitions are refined.

### 14.2 Version Model

```
SchemaVersion {
    schema_version_id: string
    clause_type_id: string
    version: int
    fields: list[Field]
    valid_from: timestamp
    valid_to: timestamp?
    changelog: string
}
```

### 14.3 Migration Strategy

When a field is renamed, split, or deprecated:

```
Field: data_retention_period (v1)
    ↓ deprecated in v2
Field: retention_policy_scope (v2)
    ↓ with migration: v1.data_retention_period → v2.retention_policy_scope
```

---

## 15. Infrastructure Differentiation

### 15.1 Five Pillars That Create a Moat

**1. Persistent Semantic Layer**

Not just embeddings (stateless) but a structured layer that stores meaning explicitly:
* Cluster → ClauseType → Fields → Intent → Policies
* Evolves over time with agent + user feedback

**2. Feature Store for Legal Signals**

Persisted, pre-computed features for fast recommendations:
* Cluster statistics, org preferences, policy violations, field distributions
* Updated incrementally via event stream

**3. Continuous Learning Pipeline**

User interaction → event logging → feature updates → model updates → better recommendations.
* Most systems: `train once → deploy → static`
* This system: `learn → improve → compound`

**4. Reasoning Infrastructure**

Structured reasoning traces stored and auditable — not LLM-generated explanations.
* Every recommendation has a trace graph linking to cluster evidence, org preferences, and policy scores.

**5. Hybrid Retrieval Layer**

Vector search + cluster filtering + metadata filtering + policy filtering — not just vector similarity.

### 15.2 What Competitors Cannot Easily Replicate

* **Feature store:** Years of accumulated clause statistics
* **Feedback loop:** User behavior → learning
* **Reasoning traces:** Structured, auditable explanation infrastructure
* **Semantic layer:** Cluster → clause → field → intent → policy mappings

---

## 16. Storage Architecture & Cost Control

### 16.1 The Problem This Section Prevents

Naive embedding storage at scale (e.g., one monolithic Elasticsearch cluster with dense_vector fields) leads to runaway costs. At 10M agreements × 8 chunks × 768-dim vectors, raw embedding storage alone reaches ~240 GB per account — before replication, indexing overhead, and segment merges. ES can inflate this to 1+ TB per account. **This design explicitly avoids that path.**

### 16.2 Tiered Storage Model

| Tier | What | Granularity | Storage per 1M Agreements | When Built | Owner |
|------|------|-------------|--------------------------|------------|-------|
| **T0: Field summaries** | 1 embedding per agreement (existing V1) | Agreement-level | ~1.5 GB | Already exists in AIDB | AIDB |
| **T1: Cluster centroids** | 1 centroid vector per micro cluster | Cluster-level | ~750 KB (~500 clusters × 1.5 KB) | After clustering | Clustering service |
| **T2: BM25 inverted index** | Per-cluster keyword index | Token-level (inverted) | ~50 MB (500 clusters × ~100 KB each) | After clustering | Clustering service |
| **T3: Chunk embeddings** | AIDB's existing chunk storage — **query, don't duplicate** | Chunk-level | 0 new bytes (already in AIDB) | Already exists | AIDB |
| **T4: Per-cluster FAISS** | Small ANN index per cluster (only for clusters with >1K chunks) | Chunk-level, partitioned | ~2 GB (20 large clusters × ~100 MB each) | On-demand, Phase 3+ | Retrieval service |

**Total new storage for 1M-agreement account: ~2–4 GB.** Not 120 GB. Not 1 TB.

### 16.3 Key Design Principle: Cluster-Partitioned Indices, Not Monolithic

Cluster-aware scoping is the primary cost control mechanism:

```
Account: 1M agreements
  → Macro clusters: ~10–20 domains (SaaS, NDA, Employment, ...)
    → ~50K–100K agreements per domain
      → Micro clusters: ~20–50 clause types per domain
        → ~2K–5K agreements per clause type
```

When retrieving similar clauses, search happens **within the relevant cluster** — not across the entire corpus. A 1M-agreement account becomes ~20 separate 50K-agreement searches. Each cluster's FAISS index is ~75 MB. Benefits:

* **Memory:** Load only the active cluster's index into memory (~75 MB), not the entire corpus
* **Latency:** ANN over 50K vectors is <5ms; ANN over 8M vectors is 50–100ms
* **Cost:** Small indices can use exact search (flat index) instead of approximate (HNSW), avoiding graph construction overhead
* **Isolation:** One cluster's index issue doesn't affect other clusters

### 16.4 What AIDB Already Stores (Don't Duplicate)

AIDB's `FieldSummaryEmbeddingsService` and `EmbeddingsService` already maintain:
* Field summary embeddings (1 per agreement, 384-dim)
* Chunk embeddings (multiple per agreement, 384-dim, ~128 tokens each)
* Chunk text (for cross-encoder reranking at query time)
* ANN search via `GetClosestEmbedding` RPC

**This design never creates a second copy of these embeddings.** All chunk-level retrieval queries route through AIDB's existing gRPC API. The only new indices are:
* Cluster centroids (tiny — for nearest-centroid assignment of new docs)
* Per-cluster BM25 indices (lightweight inverted indices for keyword matching)
* Per-cluster FAISS indices (Phase 3+, only for clusters needing sub-5ms ANN retrieval)

### 16.5 Storage Cost Comparison: This Design vs Anti-Patterns

| Approach | Storage for 1M Agreements | Monthly Cost (Azure) | Notes |
|----------|--------------------------|---------------------|-------|
| **This design (tiered, cluster-partitioned)** | ~2–4 GB new | ~$1–2/mo | Plus existing AIDB costs (unchanged) |
| Monolithic FAISS (all chunks, one index) | ~120 GB | ~$30–60/mo | Single point of failure, slow at scale |
| Elasticsearch dense_vector | ~400–800 GB (with replication + HNSW overhead) | ~$200–500/mo | ES inflates vector storage 3–5x |
| ColBERT token vectors (removed from design) | ~800 GB–1 TB | ~$300–600/mo | 100 KB per chunk × 8M chunks |

### 16.6 Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | This Design's Protection |
|-------------|-------------|-------------------------|
| **One giant vector index** | Slow ANN, high memory, single point of failure | Per-cluster FAISS indices (§16.3) |
| **Embedding + keyword search in same system (ES)** | ES does vectors badly — HNSW merge storms, shard replication of vectors, no score normalization | FAISS for vectors, lightweight BM25 lib for keywords — separate concerns |
| **Storing token-level vectors (ColBERT)** | 100 KB/chunk × millions of chunks = TB-scale storage | Cross-encoder reranker — zero precomputed storage (§11.3) |
| **Re-embedding on every schema change** | GPU cost + downtime for full re-indexing | Cluster centroids for assignment; only re-embed when switching base model (rare, planned migration) |
| **Replicating vectors across shards** | ES replicates vectors on every shard copy, multiplying storage by replica count | FAISS indices are per-cluster, single-copy, loaded on-demand from blob storage |
| **Full-corpus ANN for every query** | Slow, expensive, unnecessary when you know the cluster | Cluster-aware filtering restricts search space before ANN (§11.1) |
| **No dimensionality control** | Storing 768+ dim vectors everywhere when 384 is sufficient for most tasks | Matryoshka truncation — 384 for clustering, full dim only for precision retrieval (§3.2) |

### 16.7 Scaling Guardrails

| Guardrail | Threshold | Action |
|-----------|-----------|--------|
| Embedding count per account | >5M chunk embeddings | Alert; review if cluster-partitioned indices still fit in memory |
| Per-cluster FAISS index size | >500 MB | Split cluster or switch to IVF-PQ compressed index |
| BM25 index per cluster | >10 MB | Review tokenization; check for index bloat |
| Total new storage per account | >10 GB | Architecture review — likely means too many clusters or unnecessary indices |
| Re-embedding batch cost | >$1, 000 per run | Review if full re-embed is necessary vs incremental |

### 16.8 Migration Path: V1 → V2 Storage

```
Phase 1:  No storage changes. Operate on existing AIDB field-summary embeddings.
          New: cluster centroid vectors (~1 MB), cluster metadata in AIDB tables.

Phase 2:  Add per-cluster BM25 indices (~50 MB total).
          Evaluate Nomic v1.5 — if adopted, dual-write period:
            - New agreements get both MiniLM + Nomic embeddings in AIDB
            - Background job re-embeds historical agreements
            - Cutover when >95% re-embedded; drop old embeddings

Phase 3+: Add per-cluster FAISS indices for high-volume clusters (on-demand).
          Cross-encoder reranker deployed — no new storage.
          Contrastive fine-tuned model — replaces base model in AIDB, same storage footprint.
```

**At no phase does this design require storing >10 GB of new data per 1M-agreement account.**

---

## 17. Integration with Current Infrastructure

### 17.1 Service Bus Extensions

New queues:
* `compliance-check`: Trigger policy evaluation on agreement
* `feedback-events`: Capture user interaction signals
* `ontology-update`: Process new clause type discoveries

### 17.2 AIDB Extensions

New gRPC services:
* `OntologyService`: CRUD for ClauseType, Field, IntentDefinition
* `PolicyService`: CRUD for Policy, PolicyRule; evaluate compliance
* `PreferenceService`: Read/write org and user preferences
* `ReasoningService`: Store and retrieve recommendation traces

### 16.3 Existing Integration Points Preserved

* `FieldSummaryEmbeddingsService` → continues to provide MiniLM-384 embeddings
* `ClusteringWriteAPI` → extended to write ontology metadata
* `SchematizeHandler` → extended to write field definitions to ontology
* `BERTopicClusterer` → unchanged; new layers build on top of its output

### 16.4 Storage

| Component | Storage | Rationale |
|-----------|---------|-----------|
| Cluster artifacts | Azure Blob (existing) | Large model files |
| Ontology entities | AIDB (Postgres) | Structured queries |
| Feature store | Redis + AIDB | Fast reads + persistence |
| Reasoning traces | AIDB (Postgres) | Auditability |
| Feedback events | Service Bus → AIDB | Async ingestion |
| Vector index (ANN) | FAISS (in-memory or on-disk) | Fast retrieval |

---

## 18. Phased Implementation Plan

### Staffing & Scope Assumptions

| Assumption | Detail |
|------------|--------|
| Team size | 2–3 ML engineers + 1 backend engineer + 0.5 PM + access to legal domain reviewer |
| Domain scope (Phase 1–2) | SaaS/vendor contracts only (single contract family). Expand after validation. |
| Data quality gate | Phase 1 starts only after gold set (500 labeled summaries) is prepared and reviewed |
| Cross-functional dependency | Compliance features require legal/compliance team review before user-facing launch |
| Timeline model | Timelines are **best-case with dedicated team**. Add 30–50% buffer for enterprise review cycles, data quality issues, and cross-team dependencies. |

### Phase 1: Ontology Foundation (0–3 months)

**Extends:** `BERTopicClusterer` , `SchematizeHandler`

* **Macro clustering:** Cluster agreements into high-level domains (SaaS, NDA, Employment, Vendor) using **embedding-based summary clustering** (BERTopic on document-level summary embeddings). LLM is used only to label the resulting 5–10 macro clusters — not to classify individual documents.
* **Micro clustering:** Run BERTopic per macro domain to discover clause-level patterns
* Add `ClauseType` data model; write LLM cluster labeling pipeline
* Map each BERTopic topic_id → ClauseType with label, keywords, confidence
* Persist ontology to AIDB via new `OntologyService`
* **Universal field extraction:** Implement parallel extraction for universal fields (parties, effective date, governing law) across all agreements
* **Domain-specific extraction:** Implement cluster-conditioned prompt strategy for domain-specific fields (top 3–5 per cluster)
* **Example Set pattern:** For each cluster, select a representative sample set (~20–50 docs). Run extraction on sample set first. Present results to user for review before scaling to full cluster.
* **Field definition CRUD:** Enable users to rename, retype, redefine, add examples, create derived fields, and scope fields to clusters
* **Minimum doc threshold:** Target ~100 docs per macro cluster for stable micro-clustering. Below threshold, use LLM-only field discovery on sample set.
* **Hybrid field discovery (hackathon-validated):** Deploy Hybrid strategy (§4.4.2) as default — runs Standard + RLM in parallel, merges via 4-tier algorithm. Produces 22% more fields at highest confidence. Gracefully degrades to Standard-only if RLM times out.
* **Confidence grounding (hackathon-validated):** Replace LLM self-reported confidence with source-text grounding (§4.4.3). Stable across scale (3% degradation from 96→510 docs). Feeds into compliance gating (§6.3).
* Expose clause types in existing API responses

**What NOT to build:**
* No graphs, no graph DB
* No multi-agent system
* No contrastive learning
* No GNNs
* No compliance engine (Phase 2)
* No recommendations (Phase 3)

### Phase 2: Compliance Engine (3–6 months)

**New component:** `PolicyEngine`

* Policy definition data model (`Policy`,  `PolicyRule`)
* Compliance check pipeline: clause detection → confidence gating → field extraction → rule evaluation
* Violation reporting API with severity + confidence
* Risk dashboard data endpoints
* **Governance:** Policy rules require admin approval; violation findings require human review before action
* **Release gate:** False-positive rate ≤ 15% on audit sample before user-facing launch. Legal/compliance team sign-off required.

### Phase 3: Recommendations (6–9 months)

**New component:** `RecommendationEngine`

* Cluster frequency statistics computation
* Recommendation scoring: `α * cluster_freq + γ * policy_score`
* Similar clause retrieval within clusters (ANN search)
* Clause alternative suggestions

### Phase 4: Reasoning + Light Personalization (9–12 months)

**New components:** `ReasoningStore` , `PreferenceService`

* Reasoning trace data model and storage
* Explanation generation from traces (template + LLM)
* Organization-level preference tracking
* Feedback event capture pipeline (Service Bus → feature store)

### Phase 5: Intent + Hybrid Retrieval (12–18 months)

**New components:** `IntentLayer` , `HybridRetriever`

* Intent definitions for top clause types
* Field → intent dimension mappings
* Intent-aligned recommendation scoring
* **Validation gate:** Intent must demonstrate measurable accept-rate uplift vs Phase 3–4 baseline in A/B test before full rollout
* ColBERT-style reranking for precision retrieval
* User-level contextual personalization

### Phase 6: Evolution + Advanced (18+ months)

* Self-evolving clause discovery (anomaly → micro-cluster → agent evaluation)
* Contrastive learning pipeline
* Schema versioning
* Graph extensions (if multi-hop reasoning justified)
* Self-play evaluation (generator vs critic agents)
* Market benchmarking (cross-customer aggregation)
* Negotiation intelligence (cross-version tracking)

---

## 19. Risks & Mitigations (Engineering)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| BERTopic non-determinism across runs | High | Schema instability | Fix seeds; version cluster state; track lineage |
| LLM cluster labeling inconsistency | Medium | Ontology quality | Batch label with temperature=0; cache labels; human review for top clusters |
| Feature store staleness | Medium | Bad recommendations | TTL-based invalidation; event-driven updates |
| Cross-encoder reranker latency | Low | Slow retrieval | Only rerank top-50 post-cluster-filter; batch scoring; upgrade to distilled model if needed |
| Agent hallucinated schemas | Medium | Bad fields | Critic agent; coverage threshold (field must appear in >X% of cluster) |
| Ontology drift over time | Low-Medium | Schema confusion | Schema versioning; periodic human review |
| Cost explosion from LLM agent calls | Medium | Budget overrun | Operate agents on cluster representatives only (10–20 per cluster, not all chunks) |
| Cold-start personalization | High (early) | Weak recommendations | Fall back to industry/cluster-level stats |
| Monolithic embedding index at scale | Medium | Storage cost explosion, slow retrieval | Cluster-partitioned indices; scaling guardrails (§16.7); no vector duplication |
| RLM field discovery timeout/failure | Medium | Missing fields for 5% of clusters | Hybrid strategy with Standard fallback guarantees 100% coverage; RLM failure is gracefully handled |
| RLM latency at scale (36.5s/cluster) | Medium | Slow field discovery for large ontologies | Hybrid runs Standard + RLM in parallel; total wall-clock ≈ max(Standard, RLM) not sum. For 500 clusters: ~5 hours batch — acceptable for offline ontology construction |
| DSPy REPL sandbox escape | Low | Security risk from LLM-generated code | CodeAct sandbox restricts imports and I/O; no network access; killswitch on iteration count (25 max) and LLM calls (50 max) |
| Confidence score miscalibration | Low-Medium | False trust in extraction quality | Grounding formula uses observable signals (verbatim match, string length, numeric content), not LLM self-report; regular calibration checks against gold-labeled extractions |

---

## 20. Decision Log

| Decision | Rationale | Revisit When |
|----------|-----------|-------------|
| MiniLM-384 for Phase 1 clustering; evaluate Nomic v1.5 for Phase 2 | MiniLM is proven and integrated; Nomic offers 8K context + Matryoshka dims but must be validated on legal data | Nomic eval on gold set shows measurable improvement in cluster coherence and retrieval recall |
| No graph DB in Phase 1–4 | Relational tables + clusters sufficient | Multi-hop reasoning required |
| Cross-encoder reranker over ColBERT | Same precision, zero storage (vs ~100 KB/chunk for ColBERT), acceptable latency for top-50 candidates | Reranking latency exceeds SLO at scale, or candidate set regularly exceeds 1000 |
| Ontology from clusters (not manual) | Scalable, automatic, aligns with discovery goal | Cluster quality insufficient for ontology |
| Intent as structured layer (not embeddings) | More interpretable, auditable, controllable | Need to generalize intent to unseen clause types |
| Rule-based policies first | Simple, explainable, fast to ship | Policy complexity requires DSL or graph |
| No 2-stage clustering initially | Single BERTopic pass sufficient for Phase 1; macro→micro introduced with macro using embedding-based summary clustering (BERTopic on doc-level summary embeddings), not per-doc LLM classification | Mixed clusters degrade ontology quality; pure semantic clustering produces misleading domain groupings |
| Cluster-conditioned extraction over generic prompts | Higher precision, lower token cost, reduced hallucination | Cluster concept profiles are unreliable |
| Universal + domain-specific field separation | Reduces wasted extraction; ensures baseline fields always available | Distinction between universal and domain-specific becomes unclear |
| Conditional HyDE (Phase 3+) over always-on HyDE | Cluster-representative examples + BM25 reduce HyDE's marginal value for mature clusters; saves LLM tokens | Phase 2 A/B test shows HyDE still provides >3% F1 lift even for mature clusters |
| Composite fields as formulas over base field refs (not independent LLM extraction) | Inherits accumulated learning from base fields; deterministic arithmetic; zero LLM cost; feedback propagation through DAG | Customers need composites that require reasoning/inference beyond arithmetic (e.g., classification, summarization) |
| Tiered, cluster-partitioned storage over monolithic index | ~2–4 GB new storage per 1M accounts vs 120+ GB monolithic; avoid ES anti-patterns; query AIDB chunks instead of duplicating | Per-account storage exceeds 10 GB guardrail; single-cluster FAISS index exceeds 500 MB |
| Hybrid field discovery (Standard + RLM merge) over single approach | Hackathon eval: Hybrid produces 22% more fields (13.4 vs 11.0), highest confidence (0.751), 0% failure rate. Jaccard overlap between Standard and RLM is only 0.105 — they discover complementary fields. Marginal cost: ~$0.45 per full corpus run | RLM latency exceeds acceptable budget at scale (>60s/cluster); Standard alone achieves comparable coverage after synonym resolution |
| Source-text confidence grounding over LLM self-reported confidence | LLM confidence scores are unreliable and non-auditable. Grounded scores are stable across scale (0.725→0.704 from 96→510 docs, only 3% degradation). 42.6% of extractions score ≥0.8 (verbatim grounded) | Source-text grounding is insufficient for fields that require inference (e.g., implied terms, cross-reference resolution) |
| DSPy CodeAct REPL for RLM over multi-turn chat | CodeAct allows the LLM to write programmatic exploration code, enabling consensus voting across batched sub-LM calls. Provably explores more chunks than fixed-window approaches | DSPy CodeAct API changes or sandbox security concerns; RLM failure rate exceeds 10% |
| Text Latent Briefing (embedding-based) over KV cache Latent Briefing | API-based models (Azure OpenAI) don't expose KV cache. Embedding cosine similarity serves as an effective proxy for attention scores. Text-level approach captures the core insight (task-guided selective compression) while remaining portable. KV cache engine implemented and ready for self-hosted workers (vLLM + Qwen) | Move to self-hosted workers where KV access is available; text-level approach shows >5% quality gap vs KV-level on field discovery benchmarks |
| Latent Briefing memory-augmented RLM over stateless RLM workers | Stateless workers re-discover known fields and can't focus on gaps. Latent Briefing gives each worker a task-specific compressed view of prior reasoning, reducing redundancy. Ramp Labs reports 65% worker-token reduction and +3% accuracy. Adaptive tau adjusts compression aggressiveness based on Ramp Labs finding that harder tasks benefit from more aggressive filtering of speculative reasoning noise | Worker-to-worker memory sharing introduces latency that outweighs token savings; compaction drops critical context for edge-case clusters |

---

## 21. Implementation Backlog — Auto Field Ontology Discovery Engine

> **Epic**: FIELD-ONTOLOGY — Build a scalable auto-field ontology discovery engine for 1M+ documents
> **Owner**: APR Team
> **Target**: Replace current per-cluster-independent field identification with a unified ontology pipeline

This section provides detailed implementation tasks that realize the architecture defined in Sections 2–17. Each task includes acceptance criteria, story point estimates, and explicit cross-references to the relevant architectural sections.

---

### 21.1 EPIC 1: Foundation — Chunking & Structure

**Implements:** §2.1 (Data Contract Migration), §3.3 (Mean Pooling), §4.5 (Cluster-Conditioned Extraction)

#### FIELD-101: Replace naive text concatenation with Kamradt semantic chunking

**Priority**: P0 — Critical | **Estimate**: 5 SP | **Depends on**: None
**Cross-ref**: §2.1 (Phase 3+ chunk-level migration), §3.3 (document representation)

**Goal**: Every document is chunked into semantically coherent units with structural metadata (section headers, hierarchy level, position) before entering any downstream pipeline.

**Action Items**:
* [ ] Port `KamradtSemanticChunker` from `paragraph_clustering.py` into `src/common/chunking/`
* [ ] Add section header detection + TOC stripping (already implemented in script)
* [ ] Define `Chunk` proto/model: `{chunk_id, doc_id, text, section_heading, hierarchy_level, position, embedding}`
* [ ] Replace `"\n".join(chunks)` in `schematize_handler.py` L346 with structured chunk objects
* [ ] Add min_chunk_chars=250 guard (prevents micro-fragments)
* [ ] Write unit tests: verify chunk boundaries align with section breaks on 10 sample contracts
* [ ] Benchmark: measure chunk count reduction vs current AIDB chunking

**Acceptance Criteria**:
* Chunks preserve section boundaries (no chunk spans two sections)
* Section heading metadata attached to each chunk
* Average chunk count per document reduced by ≥40% vs current
* Token budget for field identification reduced by ≥5x

---

#### FIELD-102: Build document structure tree from chunk metadata

**Priority**: P1 — High | **Estimate**: 3 SP | **Depends on**: FIELD-101
**Cross-ref**: §4.5 (Cluster-Conditioned Extraction), §3.1 (Macro Clustering)

**Goal**: Each document has a parseable section tree (TOC) that can be shown to the LLM for field identification without sending full text.

**Action Items**:
* [ ] For each document, extract section headings + hierarchy (Article > Section > Subsection)
* [ ] Build tree structure: `{heading, level, chunk_ids, children[]}`
* [ ] Generate "synthetic TOC" string from tree (for LLM consumption)
* [ ] Store tree alongside document in AIDB/Postgres
* [ ] Add document type classifier (MSA, NDA, SLA, SOW, etc.) based on TOC + first 500 chars

**Acceptance Criteria**:
* 90%+ of section headings correctly detected on test corpus
* Synthetic TOC accurately represents document structure
* Document type classification ≥85% accuracy on labeled subset

---

#### FIELD-103: Implement TOC-first field identification prompt

**Priority**: P1 — High | **Estimate**: 3 SP | **Depends on**: FIELD-102
**Cross-ref**: §4.5.1 (HyDE Query Strategy), §16 (Cost Control)

**Goal**: Field identification LLM sees document TOCs (cheap) instead of full text (expensive), with targeted follow-up for ambiguous fields.

**Action Items**:
* [ ] Design two-round identification prompt:
  + Round 1: Send TOCs of N documents → candidate field list
  + Round 2: For uncertain fields, send 3 representative chunks → confirm/refine
* [ ] Compare field identification quality: TOC-first vs current full-text concatenation
* [ ] Measure token reduction (target: ≥7x)
* [ ] Add fallback: if TOC extraction fails, fall back to current approach

**Acceptance Criteria**:
* Field identification recall ≥95% of current approach
* Token usage per identification call reduced by ≥5x
* Latency per identification call reduced by ≥3x

---

### 21.2 EPIC 2: Scalable Clustering

**Implements:** §3.1 (Two-Level Clustering), §3.2 (Embedding Strategy), §3.4 (BERTopic Enhancements)

#### FIELD-201: Implement hierarchical two-stage clustering

**Priority**: P0 — Critical | **Estimate**: 8 SP | **Depends on**: FIELD-101
**Cross-ref**: §3.1 (Macro → Micro), §3.4 (incremental cluster assignment)

**Goal**: Cluster 1M documents by first doing coarse document-level clustering (FAISS), then fine-grained chunk-level clustering (HDBSCAN) within each partition.

**Action Items**:
* [ ] **Stage 1 — Document-level**: Compute mean-pooled document embeddings (1M × 384)
* [ ] Build FAISS IVF index (nlist=√N ≈ 1000, nprobe=10)
* [ ] Extract k-NN graph from FAISS → run Leiden community detection (or mini-batch K-Means with k=√N)
* [ ] Output: ~500-2000 coarse document clusters
* [ ] **Stage 2 — Chunk-level**: For each document cluster, run HDBSCAN on chunk embeddings
* [ ] Parallelize across CPU cores (embarrassingly parallel — one cluster per worker)
* [ ] Run BERTopic topic extraction per sub-cluster
* [ ] **Stage 3 — Topic alignment**: Merge equivalent topics across document clusters
* [ ] Write integration test: process 10K documents end-to-end, verify cluster quality
* [ ] Add memory monitoring (ensure peak RAM < 32GB for 1M docs)

**Acceptance Criteria**:
* Successfully clusters 1M documents within 4 hours on 16-core machine
* Peak memory < 32GB
* Clustering quality (silhouette score) within 5% of single-stage HDBSCAN on 10K subset
* All stages produce reproducible results (seeded randomness)

---

#### FIELD-202: Add FAISS-based document embedding index

**Priority**: P0 — Critical | **Estimate**: 5 SP | **Depends on**: None
**Cross-ref**: §3.2 (ANN search), §16 (Storage Architecture), §7.3 (Similar Contract Retrieval)

**Goal**: Persistent FAISS index for document embeddings, supporting incremental updates and nearest-neighbor queries.

**Action Items**:
* [ ] Select FAISS index type: IVF-PQ for 1M+ scale (memory-efficient)
* [ ] Build index creation pipeline: embed docs → train index → add vectors
* [ ] Implement incremental add: new docs without full rebuild
* [ ] Expose query API: `get_nearest_clusters(doc_embedding, top_k)` for document assignment
* [ ] Add serialization: save/load index to disk
* [ ] Benchmark: query latency at 1M scale (target: <10ms per doc)

**Acceptance Criteria**:
* Index build time < 30 minutes for 1M documents
* Query latency < 10ms per document
* Incremental add of 10K docs < 1 minute
* Recall@10 ≥ 0.95 vs brute-force cosine search

---

### 21.3 EPIC 3: Cross-Cluster Field Consolidation

**Implements:** §4.2 (Ontology Data Model), §4.6 (Agentic Ontology Refinement)

#### FIELD-301: Implement semantic field merging layer

**Priority**: P0 — Critical | **Estimate**: 8 SP | **Depends on**: FIELD-201
**Cross-ref**: §4.2 (ClauseType, Field models), §4.6 (Ontology Manager Agent)

**Goal**: Fields identified independently across clusters are merged into canonical fields with aliases, producing a clean deduplicated ontology.

**Action Items**:
* [ ] Collect all (field_name, definition, cluster_id) triples post-identification
* [ ] Embed each (field_name + " : " + definition) with MiniLM
* [ ] Cluster field embeddings with HDBSCAN (expected n ≈ 5K-20K)
* [ ] Within each field-cluster:
  + Pick most frequent name as canonical_name
  + Merge definitions (take longest/most informative, or LLM-merge top 3)
  + Record all aliases
* [ ] Build canonical_field → cluster_fields mapping table
* [ ] Deduplicate before extraction (don't extract same concept twice across clusters)
* [ ] Write evaluation: manually label 100 field pairs as same/different, measure merge accuracy

**Acceptance Criteria**:
* Merge precision ≥ 90% (fields merged together are truly the same concept)
* Merge recall ≥ 80% (duplicate fields across clusters are actually merged)
* Ontology size reduced by ≥ 10x vs raw per-cluster field count
* Each canonical field has ≤ 1 definition (conflicts resolved)

---

#### FIELD-302: Build field alias resolution table

**Priority**: P1 — High | **Estimate**: 3 SP | **Depends on**: FIELD-301
**Cross-ref**: §4.2 (Field data model), §14 (Schema Versioning)

**Goal**: Persistent lookup table mapping all field name variants to canonical field IDs, usable at query time and in extraction pipelines.

**Action Items**:
* [ ] Store alias table in Postgres: `{alias_name, canonical_field_id, confidence, source_cluster_ids}`
* [ ] Add fuzzy lookup API: given a field name, return canonical field + confidence
* [ ] Populate from FIELD-301 output
* [ ] Add admin UI for manual alias overrides
* [ ] Version the alias table (track changes over time)

**Acceptance Criteria**:
* Lookup latency < 5ms
* Handles case/whitespace/punctuation normalization
* Manual overrides persist across pipeline re-runs

---

### 21.4 EPIC 4: Fix Wiring Bugs & Defaults

**Implements:** Bug fixes in current `SchematizeHandler` and selector infrastructure

#### FIELD-401: Pass representative_keywords from clustering to field identification

**Priority**: P0 — Critical (quick win) | **Estimate**: 1 SP | **Depends on**: None
**Cross-ref**: §4.5 (Cluster-Conditioned Extraction)

**Goal**: Fix the hardcoded `representative_keywords=[]` so the field identification LLM receives cluster topic keywords.

**Action Items**:
* [ ] In `schematize_handler.py` L692, replace `representative_keywords=[]` with actual keywords from BERTopic topic representation
* [ ] Verify `Cluster` model carries `representative_keywords` from clustering output
* [ ] If not available, extract top-10 c-TF-IDF keywords from cluster's documents
* [ ] Add integration test: verify keywords appear in AIFlow request payload
* [ ] A/B test: compare field identification quality with vs without keywords

**Acceptance Criteria**:
* `representative_keywords` is non-empty for all clusters with ≥ 5 documents
* Field identification prompt receives keywords
* No regression in field identification quality

---

#### FIELD-402: Switch default subset selector to DPP

**Priority**: P1 — High (quick win) | **Estimate**: 1 SP | **Depends on**: None
**Cross-ref**: §4.5 (Cluster-Conditioned Extraction representative selection)

**Goal**: Use DPP (Determinantal Point Process) for identification subset selection to ensure diversity, not just top membership probability.

**Action Items**:
* [ ] Change `DEFAULT_SUBSET_STRATEGY` from `"top_membership"` to `"dpp"` in constants.py
* [ ] Verify DPP code path works with current embeddings (it exists but is untested in production)
* [ ] Add fallback: if DPP fails (missing embeddings), fall back to `weighted_random`
* [ ] Increase `SELECTOR_DEFAULT_MAX_DOCUMENTS` from 50 → 100 for identification stage
* [ ] A/B test: compare field discovery breadth (unique fields found) with DPP vs TopMembership

**Acceptance Criteria**:
* DPP runs without errors on all existing clusters
* Field identification discovers ≥ 10% more unique fields on diverse clusters
* Fallback to weighted_random activates correctly when embeddings unavailable

---

#### FIELD-403: Add stratified sub-sampling for large clusters

**Priority**: P2 — Medium | **Estimate**: 3 SP | **Depends on**: FIELD-402
**Cross-ref**: §3.1 (micro clustering), §18 Phase 1 (minimum doc threshold)

**Goal**: For clusters > 500 documents, run mini-HDBSCAN to find sub-groups and sample proportionally from each, ensuring tail/edge documents are represented.

**Action Items**:
* [ ] In DPPSelector, add cluster-size threshold check (> 500 docs)
* [ ] Run lightweight HDBSCAN (min_cluster_size=max(5, N/100)) within the cluster
* [ ] Sample proportionally from each sub-group (maintain DPP within sub-groups)
* [ ] Pass sub-group labels as metadata to field identification ("this doc is from sub-group X of Y")
* [ ] Evaluate: measure field discovery on synthetic mixed clusters (90% MSA + 10% vendor agreements)

**Acceptance Criteria**:
* Correctly identifies fields present only in minority sub-groups (≥ 80% recall)
* No increase in identification latency for clusters < 500 docs
* Sub-group computation < 5 seconds for 2000-doc clusters

---

### 21.5 EPIC 5: Retrieval Overhaul

**Implements:** §11 (Hybrid Retrieval Layer), §4.5.1 (Conditional HyDE)

#### FIELD-501: Replace AIDB vector search with local BM25 + dense hybrid retrieval

**Priority**: P0 — Critical | **Estimate**: 8 SP | **Depends on**: FIELD-101
**Cross-ref**: §11.2 (Hybrid Retrieval Architecture), §7.3 (Similar Clause Retrieval)

**Goal**: Eliminate all per-field per-document AIDB round-trips. Build local indexes per cluster and query all fields in batch.

**Action Items**:
* [ ] Port `SparseRetriever` (BM25) from `paragraph_clustering.py` into `src/common/retrieval/`
* [ ] Port `DenseRetriever` (cosine) into same module
* [ ] Implement reciprocal rank fusion (RRF): `score = Σ 1/(k + rank_i)` across BM25 + dense
* [ ] Build per-cluster index at cluster processing time (BM25 corpus + FAISS for chunk embeddings)
* [ ] Replace `aidb_client.get_closest_embedding()` calls with local hybrid retrieval
* [ ] Batch all field queries against the same index (15 fields × 1 index = 15 local queries, not 150 AIDB calls)
* [ ] Benchmark: latency, precision, recall vs current AIDB approach on test cluster

**Acceptance Criteria**:
* Snippet retrieval latency reduced by ≥ 100x (from ~500ms/query to <5ms/query)
* Retrieval quality (nDCG@3) ≥ 95% of current AIDB approach
* Zero external API calls for snippet retrieval
* All fields for a cluster retrieved in < 1 second total

---

#### FIELD-502: Add ColBERT re-ranker for nuanced field queries

**Priority**: P2 — Medium | **Estimate**: 5 SP | **Depends on**: FIELD-501
**Cross-ref**: §11.2 (Cross-Encoder Reranker), §3.2 (Hybrid retrieval architecture)

**Goal**: For fields where BM25 + dense retrieval disagree (low confidence), apply ColBERT MaxSim re-ranking on the top-50 candidates for higher precision.

**Action Items**:
* [ ] Port `LateInteractionRetriever` from `paragraph_clustering.py` into `src/common/retrieval/`
* [ ] Make ColBERT re-ranking optional, triggered when:
  + BM25 top-1 and dense top-1 differ
  + Top-1 score is below confidence threshold
  + Field query is abstract (low keyword overlap expected)
* [ ] Build token-level index only for top-50 candidates (not full corpus — saves memory)
* [ ] Benchmark: measure precision lift on fields where BM25/dense disagree

**Acceptance Criteria**:
* Precision@3 improvement ≥ 5% on ambiguous field queries vs BM25-only
* ColBERT re-ranking adds < 100ms per field
* Token index memory bounded to top-50 chunks (not full corpus)

---

#### FIELD-503: Remove HyDE dependency — use raw field queries with hybrid retrieval

**Priority**: P1 — High | **Estimate**: 2 SP | **Depends on**: FIELD-501
**Cross-ref**: §4.5.1 (Conditional HyDE — "Make HyDE conditional" decision)

**Goal**: Eliminate all HyDE LLM calls. Use raw `"field_name: definition"` as the retrieval query directly against BM25 + dense hybrid index.

**Rationale** (from retrieval comparison on 10 docs / 198 chunks / 14 field queries):
* BM25 with raw keyword queries: Precision@5=1.0, MRR=1.0, query time=0.6ms
* Dense cosine with raw queries: Precision@5=0.986, MRR=1.0, query time=474ms
* ColBERT with raw queries: Precision@5=1.0, nDCG@10=0.986, query time=931ms
* HyDE adds 1 LLM call + 1 embedding call per field per cluster — thousands of calls for zero precision lift

**Action Items**:
* [ ] Remove `hyde_query_prompt.txt` usage from `aidb_snippet_selector.py` extraction path
* [ ] Remove `execute_generic_openai()` HyDE generation call from `extractions.py` L55-65
* [ ] Replace query construction with: `query = f"{field_name}: {definition}"` (raw text)
* [ ] Pass raw query to hybrid retrieval (FIELD-501) instead of HyDE-embedded query to AIDB
* [ ] A/B test on 5 clusters: raw query hybrid retrieval vs HyDE+AIDB cosine
* [ ] Keep HyDE as opt-in fallback (env flag `ENABLE_HYDE=false`) for fields with <50% BM25 recall
* [ ] Delete or archive `prompts/hyde_query_prompt.txt`

**Acceptance Criteria**:
* Zero LLM calls for snippet retrieval in default path
* Retrieval quality (nDCG@3) ≥ current HyDE+AIDB baseline
* Per-field retrieval latency reduced from ~700ms (LLM+embed+AIDB) to <5ms (local hybrid)
* HyDE fallback activatable via config for edge cases

---

### 21.6 EPIC 6: Adaptive Extraction Budget

**Implements:** §4.5 (Cluster-Conditioned Extraction), §6.3 (Confidence Gating)

#### FIELD-601: Implement confidence-gated extraction expansion

**Priority**: P1 — High | **Estimate**: 5 SP | **Depends on**: FIELD-501
**Cross-ref**: §6.3 (Confidence Gating thresholds), §4.3 (Universal vs Domain-Specific Fields)

**Goal**: Replace the hard cap of 10 extraction docs with an adaptive system that expands sampling for high-variance fields and skips LLM for stable fields.

**Action Items**:
* [ ] **Phase 1 — Discovery extraction**: Keep Cochran sample (cap at 10), extract for all fields
* [ ] **Phase 2 — Measure agreement**: For each field across 10 docs:
  + Compute extraction success rate (found / attempted)
  + Compute value entropy (unique values / attempts)
  + Compute extraction confidence (model logprob if available, else heuristic)
* [ ] **Phase 3 — Adaptive expansion**:
  + High agreement (>0.9) + low entropy → stable field → rules/regex extraction for remaining docs
  + High agreement + high entropy → variable field → expand to 50 docs with LLM
  + Low agreement (<0.5) → questionable field → flag for review, don't expand
* [ ] Remove hardcoded `DOC_FIELD_EXTRACTION_MAX_NUMBER = 10` cap
* [ ] Add per-field extraction budget config: `{field_id: max_docs, strategy}`
* [ ] Track extraction cost per field for budget optimization

**Acceptance Criteria**:
* Total LLM extraction calls reduced by ≥ 50% vs extracting 10 docs for all fields
* Stable fields (Governing Law, Effective Date) detected and switched to regex within first run
* Variable fields get ≥ 3x more sample docs than current fixed-10 approach
* No reduction in extraction quality (measured on labeled test set)

---

#### FIELD-602: Implement batch extraction (multiple snippets per LLM call)

**Priority**: P1 — High | **Estimate**: 3 SP | **Depends on**: FIELD-501
**Cross-ref**: §16 (Cost Control), §4.5 (extraction pipeline)

**Goal**: Instead of 1 LLM call per field per document, batch 10-20 snippets (from different documents) into a single extraction call for the same field.

**Action Items**:
* [ ] Design batch extraction prompt: "Extract [Field Name] from each of these snippets: [1] ... [2] ... [3] ..."
* [ ] Implement response parser: map extracted values back to source documents
* [ ] Handle partial failures (some snippets don't contain the field)
* [ ] Determine optimal batch size (test 5, 10, 20 snippets per call)
* [ ] Benchmark: extraction quality at batch-10 vs individual calls

**Acceptance Criteria**:
* LLM calls reduced by 10x (batch size = 10)
* Extraction F1 within 2% of individual-call baseline
* Latency per-field-per-cluster reduced by ≥ 5x

---

### 21.7 EPIC 7: Ontology Construction

**Implements:** §4 (Ontology Layer), §5 (Intent Layer)

#### FIELD-701: Build field co-occurrence matrix and PMI scores

**Priority**: P1 — High | **Estimate**: 3 SP | **Depends on**: FIELD-301, FIELD-601
**Cross-ref**: §5.2 (Intent Schema), §4.2 (Field data model)

**Goal**: Compute which fields co-occur across documents, enabling relationship discovery and field hierarchy construction.

**Action Items**:
* [ ] Build binary doc × field matrix from extraction results
* [ ] Compute pointwise mutual information (PMI) for all field pairs
* [ ] Identify:
  + **Co-occurring** (high PMI): "Indemnification" + "Limitation of Liability"
  + **Mutually exclusive** (negative PMI): "Fixed Price" vs "Time & Materials"
  + **Implication** (asymmetric): if "SLA Penalties" then almost always "Service Levels"
* [ ] Store co-occurrence data in Postgres: `{field_a, field_b, pmi, co_count, a_count, b_count}`
* [ ] Add API: `get_related_fields(field_id)` → ranked related fields

**Acceptance Criteria**:
* PMI computed for all field pairs in < 10 minutes for 1M-doc corpus
* Top-5 related fields for each canonical field are semantically meaningful (manual review)
* API response time < 50ms

---

#### FIELD-702: Build hierarchical field taxonomy

**Priority**: P2 — Medium | **Estimate**: 5 SP | **Depends on**: FIELD-701
**Cross-ref**: §4.2 (parent_clause_type_id), §5.3 (Intent Construction)

**Goal**: Automatically organize fields into a 3-4 level hierarchy (e.g., Legal Provisions → Indemnification → Cap on Indemnity).

**Action Items**:
* [ ] Combine field embedding similarity + PMI co-occurrence into composite distance
* [ ] Run agglomerative clustering on fields → dendrogram
* [ ] Cut at 3-4 levels to produce hierarchy: Category → Subcategory → Field → Subfield
* [ ] Auto-label categories using LLM (given child fields, generate parent label)
* [ ] Compute field prevalence: Universal (>90%), Common (50-90%), Occasional (10-50%), Rare (<10%)
* [ ] Export as versioned ontology JSON

**Acceptance Criteria**:
* Hierarchy has 3-4 levels with 8-15 top-level categories
* Manual review: ≥ 85% of parent-child relationships are semantically correct
* Prevalence labels accurately reflect document corpus statistics

---

#### FIELD-703: Build field relationship graph and suggestions API

**Priority**: P3 — Low | **Estimate**: 3 SP | **Depends on**: FIELD-701, FIELD-702
**Cross-ref**: §13 (Graph-Based Extensions), §7 (Recommendation Engine)

**Goal**: Graph-based API that suggests related fields ("You added Indemnification — consider also adding Limitation of Liability").

**Action Items**:
* [ ] Build networkx graph: nodes=fields, edges=(co-occurrence, hierarchy, implication)
* [ ] Export to JSON format for frontend consumption
* [ ] Implement suggestion API: `suggest_fields(selected_fields)` → ranked recommendations
* [ ] Add "required fields" suggestions based on document type + prevalence
* [ ] Visualize graph for internal exploration (pyvis or similar)

**Acceptance Criteria**:
* Suggestions are relevant ≥ 80% of the time (manual evaluation)
* API latency < 100ms
* Graph export < 1MB for typical ontology (500 fields)

---

### 21.8 EPIC 8: Incremental Processing

**Implements:** §3.4 (Incremental cluster assignment), §14 (Schema Versioning)

> **Critical for production viability.** Full re-clustering is expensive and cannot run on every new document. Incremental assignment is required for real-time workflows and must ship in Phase 2.

#### FIELD-801: Implement document-to-cluster assignment for new documents

**Priority**: P0 — Critical | **Estimate**: 5 SP | **Depends on**: FIELD-202
**Cross-ref**: §3.4 (incremental cluster assignment via nearest-centroid)

**Goal**: New documents are assigned to existing clusters without re-clustering the entire corpus.

**Action Items**:
* [ ] On ingestion: embed new document → query FAISS → find top-5 nearest cluster centroids
* [ ] Assignment rules:
  + max_sim > 0.85 → assign to best cluster, run extraction pipeline only
  + 0.6 < max_sim < 0.85 → assign tentatively, add to review queue
  + max_sim < 0.6 → add to "novel documents" buffer
* [ ] Run extraction for assigned document using existing cluster's field definitions
* [ ] Update cluster centroid (running mean) after assignment
* [ ] Track assignment confidence in metrics

**Acceptance Criteria**:
* New document assignment < 500ms (embed + FAISS query)
* Assignment accuracy ≥ 90% (verified against re-clustering ground truth)
* Novel document buffer correctly captures genuinely new document types

---

#### FIELD-802: Implement novel document buffer and periodic re-clustering

**Priority**: P2 — Medium | **Estimate**: 5 SP | **Depends on**: FIELD-801
**Cross-ref**: §3.4 (cluster drift detection), §14 (Schema Versioning)

**Goal**: Documents that don't fit existing clusters accumulate in a buffer and are periodically re-clustered, with results merged into the existing ontology.

**Action Items**:
* [ ] Implement buffer with configurable threshold (default: 500 docs)
* [ ] When buffer threshold reached: run full clustering on buffer documents only
* [ ] Compare new clusters against existing clusters (centroid similarity)
* [ ] If new cluster matches existing: merge (update centroid, add docs)
* [ ] If genuinely new: create new cluster, run field identification, merge fields into ontology
* [ ] Add ontology versioning: each merge creates a new version with changelog
* [ ] Add cluster drift detection: alert when centroid moves > threshold

**Acceptance Criteria**:
* Buffer processing < 30 minutes for 500 documents
* New clusters correctly merged or created (manual review on test data)
* Ontology versioning produces clean diff between versions
* Drift alert triggers within 1 re-clustering cycle of actual drift

---

### 21.9 EPIC 9: Quality & Confidence

**Implements:** §6.3 (Confidence Gating), §11.1 (Governance Operating Model)

#### FIELD-901: Implement composite field confidence scoring

**Priority**: P1 — High | **Estimate**: 5 SP | **Depends on**: FIELD-301, FIELD-601
**Cross-ref**: §6.3 (confidence gating thresholds), §9 (Evaluation Framework)

**Goal**: Every field in the ontology has a confidence score that determines whether it's auto-included, sent for review, or discarded.

**Action Items**:
* [ ] Compute 4 sub-scores per field:
  1. **Discovery confidence**: # clusters that independently identified this field / total clusters
  2. **Extraction success rate**: extracted / attempted across Cochran samples
  3. **Value consistency**: 1 - normalized entropy of extracted values
  4. **Definition stability**: mean cosine similarity of definitions across clusters
* [ ] Combine into composite: `score = α₁×discovery + α₂×extraction + α₃×consistency + α₄×stability`
* [ ] Set initial weights: α₁=0.3, α₂=0.3, α₃=0.2, α₄=0.2
* [ ] Apply thresholds:
  + Score > 0.8 → auto-include in ontology
  + 0.5-0.8 → human review queue
  + < 0.5 → auto-discard
* [ ] Store scores in field metadata, expose in API
* [ ] Build leaderboard dashboard: fields ranked by confidence

**Acceptance Criteria**:
* All fields have composite scores
* Auto-included fields are ≥ 95% valid (manual review of sample)
* Auto-discarded fields are ≥ 90% correctly rejected
* Review queue is manageable size (< 20% of total fields)

---

#### FIELD-902: Build human-in-the-loop review queue

**Priority**: P2 — Medium | **Estimate**: 5 SP | **Depends on**: FIELD-901
**Cross-ref**: §11.1 (Ontology Governance), §11.2 (Compliance Governance)

**Goal**: Fields with confidence 0.5-0.8 are presented to reviewers who can approve, reject, merge, or edit them — and their decisions improve future scoring.

**Action Items**:
* [ ] Build review queue UI: show field name, definition, aliases, confidence breakdown, sample extractions
* [ ] Reviewer actions: Approve, Reject, Merge with (pick canonical), Edit definition
* [ ] Store reviewer decisions as training signal
* [ ] After sufficient decisions (~200), retrain confidence weights (logistic regression on reviewer labels)
* [ ] Track inter-reviewer agreement for quality control

**Acceptance Criteria**:
* Reviewer can process 50 fields/hour
* Confidence model improves by ≥ 5% accuracy after retraining
* All reviewer decisions are auditable (who, when, what)

---

### 21.9.5 EPIC 10: V2 Core Architecture Implementation

**Implements:** §3.5 (Clause-Level Semantic Embeddings), §4.7 (Hierarchical Topic Merging & Synonym Resolution)

> **These are P0 critical implementation items, not exploratory spikes.** The architectural changes in this epic are fundamental to V2's value proposition and must be completed in Phase 1–2. Each item produces:
> 1. Production-ready code integrated into the main pipeline
> 2. Quantitative validation against success criteria
> 3. Full migration of existing data

#### FIELD-110: Clause-Level Semantic Embeddings

**Priority**: P0 — Critical | **Estimate**: 5 SP | **Depends on**: FIELD-101
**Cross-ref**: §3.5 (Clause-Level Semantic Embeddings), §2.1 (Data Contract Migration)

**Goal**: Replace field-summary embeddings with clause-level (per-chunk) embeddings as the primary input signal for clustering.

**Action Items**:
* [ ] Implement `ClauseLevelEmbedder` class with context window support
* [ ] Integrate into main clustering pipeline (not parallel/shadow)
* [ ] Run BERTopic on clause-level embeddings
* [ ] Validate cluster quality metrics: silhouette score, intra-cluster cosine similarity, keyword coherence
* [ ] Complete migration: all new agreements use clause-level embeddings
* [ ] Backfill: re-embed existing agreements (batch job)
* [ ] Document storage/compute cost and operational runbook

**Evaluation Metrics**:
| Metric | Baseline (Field Summary) | Target (Clause-Level) |
|--------|-------------------------|----------------------|
| Silhouette score | Baseline | +0.05 |
| Cluster purity (manual review) | Baseline | +10% |
| Intra-cluster similarity | Baseline | +0.03 |

**Acceptance Criteria**:
* Clause-level embeddings integrated as default in production pipeline
* Cluster purity improved by at least 5% (manual review of 100 clusters)
* Migration complete for >95% of agreements
* Storage cost increase documented and within bounds (<2x)

---

#### FIELD-111: Field Processor Performance Improvements

**Priority**: P0 — Critical | **Estimate**: 3 SP | **Depends on**: FIELD-110, FIELD-501
**Cross-ref**: §3.5.4 (Field Processor Performance), §16 (Cost Control)

**Goal**: Implement batched extraction, prompt caching, and async processing to reduce extraction latency by 5x+ and token cost by 30%+.

**Action Items**:
* [ ] Implement `BatchClauseExtractor` with configurable batch size
* [ ] Add prompt caching layer (cluster_type → compiled prompt)
* [ ] Implement async parallel extraction with rate limiting
* [ ] Add early termination for high-confidence stable fields (regex fallback)
* [ ] Benchmark on 100-document test cluster:
  + Measure extraction latency (end-to-end)
  + Measure LLM API calls (count and tokens)
  + Measure extraction quality (F1 on labeled subset)
* [ ] Deploy to production and monitor metrics

**Target Metrics**:
| Metric | Current | Target |
|--------|---------|--------|
| Extraction latency (100 docs) | ~5 min | <1 min |
| LLM calls per cluster | N × F | N/10 × F |
| Token consumption | Baseline | -30% |
| Extraction F1 | Baseline | No regression |

**Acceptance Criteria**:
* 5x latency improvement achieved in production
* 30% token reduction achieved in production
* No regression in extraction quality (F1 within 2% of baseline)

---

#### FIELD-310: Hierarchical Topic Merging

**Priority**: P0 — Critical | **Estimate**: 5 SP | **Depends on**: FIELD-201, FIELD-301
**Cross-ref**: §4.7 (Hierarchical Topic Merging), §3.4 (BERTopic Enhancements)

**Goal**: Implement automatic topic merging to reduce cluster fragmentation and create hierarchical ontology structure.

**Action Items**:
* [ ] Implement `HierarchicalTopicMerger` class with configurable similarity threshold
* [ ] Integrate into main clustering pipeline as post-processing step
* [ ] Tune merge threshold based on precision/recall tradeoff (target: 0.85)
* [ ] Implement hierarchy detection: identify parent-child relationships within large merged clusters
* [ ] Store merge provenance and aliases in ontology data model
* [ ] Build UI for merge visualization in ontology viewer
* [ ] Deploy to production

**Evaluation Metrics**:
| Metric | Target |
|--------|--------|
| Merge precision | ≥90% |
| Merge recall | ≥80% |
| Cluster reduction | 15-30% |
| Ontology coherence (user rating) | Improved |

**Acceptance Criteria**:
* Merge precision ≥90% in production
* Cluster count reduced by at least 15%
* Ontology coherence maintained or improved (user feedback)
* Hierarchy visualization deployed in UI

---

#### FIELD-311: Synonym Resolution System

**Priority**: P0 — Critical | **Estimate**: 3 SP | **Depends on**: FIELD-301, FIELD-310
**Cross-ref**: §4.7.3 (Synonym Resolution), §4.2 (Field data model)

**Goal**: Implement field synonym resolution to unify field names across clusters and reduce extraction duplication.

**Action Items**:
* [ ] Implement `SynonymResolver` class with known synonym lookup and embedding fallback
* [ ] Seed known synonym table with 100+ legal domain field groups
* [ ] Integrate into extraction pipeline: resolve field names before storage
* [ ] Implement reverse lookup API: given canonical, return all variants
* [ ] Build admin UI for synonym table management
* [ ] Deploy to production

**Evaluation Metrics**:
| Metric | Target |
|--------|--------|
| Known synonym coverage | 70%+ of common fields |
| Resolution precision | ≥95% (known) / ≥85% (embedding) |
| Field deduplication rate | 20-40% reduction |

**Acceptance Criteria**:
* Known synonym lookup covers at least 70% of fields in production
* Resolution precision ≥85% on evaluation set
* Field deduplication rate 20-40% achieved
* Admin UI deployed for synonym management

---

### 21.10 Priority & Phasing Summary

| Phase | Epics | Items | Goal | Maps to §18 Phase |
|-------|-------|-------|------|-------------------|
| **Phase 0: Quick Wins** (Week 1-2) | 4 | FIELD-401, FIELD-402 | Fix wiring bugs, switch to DPP — immediate quality lift | Pre-Phase 1 |
| **Phase 1: Foundation + Core V2** (Week 3-6) | 1, 5, 10 | FIELD-101, FIELD-102, FIELD-501, FIELD-503, **FIELD-110, FIELD-310** | Kamradt chunking + hybrid retrieval + **clause embeddings + topic merging** | Phase 1 |
| **Phase 2: Scale + Performance** (Week 7-12) | 2, 3, 8, 10 | FIELD-201, FIELD-202, FIELD-301, FIELD-302, **FIELD-111, FIELD-311, FIELD-801** | Hierarchical clustering + field merging + **perf + synonyms + incremental assignment** | Phase 1-2 |
| **Phase 3: Efficiency** (Week 13-16) | 6 | FIELD-601, FIELD-602, FIELD-103 | Adaptive extraction + batch LLM + TOC-first identification | Phase 2-3 |
| **Phase 4: Intelligence** (Week 17-22) | 7, 9 | FIELD-701, FIELD-702, FIELD-901, FIELD-902 | Ontology construction + confidence scoring + human review | Phase 3-4 |
| **Phase 5: Continuous** (Week 23-26) | 8 | FIELD-802 | Novel buffer + periodic re-clustering + drift detection | Phase 4-5 |
| **Phase 6: Polish** (Week 27+) | 5, 7 | FIELD-502, FIELD-403, FIELD-703 | ColBERT re-ranker, stratified sampling, field suggestions | Phase 5-6 |

---

### 21.11 Dependency Graph

```
                   FIELD-101 (Kamradt chunking)
                  /          |            \
          FIELD-102      FIELD-501      FIELD-201
         (doc tree)    (hybrid retr)   (hier. cluster)
             |              |              |
          FIELD-103      FIELD-502      FIELD-202
         (TOC-first)    (ColBERT)     (FAISS index)
                          |              |
                       FIELD-503      FIELD-301
                    (kill HyDE)    (field merge)
                                       |
                   ┌───────────────FIELD-302────────────────┐
                   |            (alias table)               |
                FIELD-601                              FIELD-701
             (adaptive extr)                         (co-occurrence)
                   |                                       |
                FIELD-602                              FIELD-702
             (batch extract)                          (hierarchy)
                                                          |
                FIELD-901                              FIELD-703
             (confidence)                          (suggestions)
                   |
                FIELD-902
             (review queue)

 V2 Core Architecture (EPIC 10 — P0 Critical):
 ─────────────────────────────────────────────
 FIELD-101 ──→ FIELD-110 (clause embeddings) ──→ FIELD-111 (perf improvements)

 FIELD-201 + FIELD-301 ──→ FIELD-310 (topic merging) ──→ FIELD-311 (synonym resolution)

 Incremental Processing (EPIC 8 — P0 Critical):
 ─────────────────────────────────────────────
 FIELD-202 (FAISS) ──→ FIELD-801 (incremental assignment) ──→ FIELD-802 (novel buffer)

 Independent: FIELD-401, FIELD-402 (quick wins, start immediately)
```

---

### 21.12 Total Estimates

| Category | Story Points | Items |
|----------|-------------|-------|
| P0 Critical | 44 SP | 10 items (FIELD-101, 110, 111, 201, 202, 301, 310, 311, 401, **801**) |
| P1 High | 22 SP | 6 items (FIELD-102, 103, 402, 501, 503, 601, 602, 701, 901) |
| P2 Medium | 18 SP | 5 items (FIELD-403, 502, 702, 802, 902) |
| P3 Low | 3 SP | 1 item (FIELD-703) |
| **Total** | **87 SP** | **22 items** |

---

### 21.13 Cross-Reference Index

| JIRA Task | Primary Section Reference | Secondary References |
|-----------|--------------------------|---------------------|
| FIELD-101 | §2.1 Data Contract Migration | §3.3, §4.5 |
| FIELD-102 | §4.5 Cluster-Conditioned Extraction | §3.1 |
| FIELD-103 | §4.5.1 HyDE Query Strategy | §16 |
| FIELD-201 | §3.1 Two-Level Clustering | §3.4 |
| FIELD-202 | §3.2 Embedding Strategy | §16, §7.3 |
| FIELD-301 | §4.2 Ontology Data Model | §4.6 |
| FIELD-302 | §4.2 Field data model | §14 |
| FIELD-401 | §4.5 Cluster-Conditioned Extraction | — |
| FIELD-402 | §4.5 Representative selection | — |
| FIELD-403 | §3.1 Micro clustering | §18 Phase 1 |
| FIELD-501 | §11.2 Hybrid Retrieval | §7.3 |
| FIELD-502 | §11.2 Cross-Encoder Reranker | §3.2 |
| FIELD-503 | §4.5.1 Conditional HyDE | — |
| FIELD-601 | §6.3 Confidence Gating | §4.3 |
| FIELD-602 | §16 Cost Control | §4.5 |
| FIELD-701 | §5.2 Intent Schema | §4.2 |
| FIELD-702 | §4.2 parent_clause_type_id | §5.3 |
| FIELD-703 | §13 Graph-Based Extensions | §7 |
| FIELD-801 | §3.4 Incremental assignment | — |
| FIELD-802 | §3.4 Cluster drift | §14 |
| FIELD-901 | §6.3 Confidence Gating | §9 |
| FIELD-902 | §11.1 Ontology Governance | §11.2 |
| **FIELD-110** | **§3.5 Clause-Level Semantic Embeddings** | §2.1, §3.2 |
| **FIELD-111** | **§3.5 Field Processor Performance** | §4.5, §16 |
| **FIELD-310** | **§4.7 Hierarchical Topic Merging** | §3.4, §4.2 |
| **FIELD-311** | **§4.7 Synonym Resolution** | §4.2, §4.6 |
