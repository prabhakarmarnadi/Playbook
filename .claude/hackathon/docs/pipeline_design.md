# Field Discovery & Extraction Pipeline

## The Problem Today

The current pipeline is entangled inside the field schematizer with tightly-coupled dependencies [^max-eval-decouple]. Snippet selection using the aging MiniLM-L12-v2 (23M params, 384-dim) is the primary quality bottleneck — not the extraction LLM. Swapping embedding models from MiniLM to nomic-embed-text-v1.5 yields +6.3% F1 on retrieval (70.0% → 76.3%), while swapping extraction LLMs yields <4% delta [^max-eval-embeddings]. Context rot on long documents causes cascading failures across date, renewal, and termination fields [^max-eval-context-rot].

The clustering algorithm also needs upgrading. EVoC (Embedding Vector Oriented Clustering) replaces the UMAP + HDBSCAN two-step with a single cosine-native pass — no dimensionality reduction needed. On the same dataset it delivers higher quality (0.920 vs 0.908), 38% faster runtime, and only 4 tunable hyperparameters vs 6 [^aaditya-eval-evoc].

---

## Pipeline 1: Field Discovery (primary work)

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INGEST & EMBED                                            │
│                                                                     │
│  Documents (PDF/text)                                               │
│       │                                                             │
│       ├─→ Embedding chunks (2000 tok, 100 overlap)                 │
│       │     Used by: clustering, RAG retrieval, dedup               │
│       │     chunk_type="embedding"                                   │
│       ├─→ Extraction chunks (4000 tok, structural boundaries)      │
│       │     Used by: Pipeline 3 sequential extraction               │
│       │     chunk_type="extraction"                                  │
│       ├─→ Clause chunks (structural split → ~20-30 per doc)        │
│       │     Used by: Pipeline 2 clause-level clustering             │
│       │     chunk_type="clause"                                      │
│       │     Split at contractual concept boundaries                  │
│       │     (~2,000 valid clauses from 98 docs in eval)             │
│       ├─→ NL Index (TOC/heading structure via 5.4-nano, one-time)  │
│       └─→ Embeddings per chunk (nomic-embed-text-v1.5)             │
│            Embedding + clause chunks get vectors; extraction don't  │
│                                                                     │
│  Store: pgvector collection per account                             │
│  All chunk types stored in same schema, tagged by chunk_type        │
│  Why nomic: 76.3% retrieval F1 vs 70% MiniLM, Apache license,     │
│  independently validated by both evals on different tasks           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ STAGE 2: DOCUMENT CLUSTERING (EVoC + Optuna)                       │
│                                                                     │
│  Input: document-level embeddings (mean-pool over chunks)           │
│       │                                                             │
│       ├─→ EVoC clustering on cosine distance (no UMAP)             │
│       ├─→ Scoring-based layer selection                             │
│       │     (silhouette + cosine + outlier ratio)                   │
│       ├─→ 30-trial Optuna optimization (4 params only)             │
│       │     base_min_cluster_size, noise_level,                     │
│       │     merge_threshold, layer scoring weights                  │
│       ├─→ Near-duplicate cluster merge (0.96 cosine threshold)     │
│       └─→ Outlier assignment to nearest cluster                     │
│                                                                     │
│  Output: ~10 document domains for hierarchical navigation           │
│  Quality: 0.920 avg (highest tested)                                │
│  Speed: ~160s clustering, 6x faster than HDBSCAN                   │
│  dynamic_scaler.py adjusts EVoC params for 500→10k docs            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                     ┌─────────▼─────────┐
                     │  FEATURE FLAG      │
                     │  discovery_backend │
                     └────┬─────────┬────┘
                          │         │
       ┌──────────────────▼──┐  ┌───▼─────────────────────────────────┐
       │ "standard" (legacy)  │  │ "hybrid" (default)                  │
       │                      │  │                                     │
       │ STAGE 3a: FIELD      │  │ STAGE 3b: HYBRID FIELD DISCOVERY   │
       │ DISCOVERY (legacy)   │  │ (per cluster)                       │
       │                      │  │                                     │
       │ Subset: α·√N docs   │  │ Subset: α·√N docs (TopMembership), │
       │ (TopMembership),     │  │ max 50. 15 fields max per cluster, │
       │ max 50               │  │ confidence filtering                │
       │ 15 fields max,       │  │                                     │
       │ 5 async workers,     │  │ Two methods run in parallel, merge: │
       │ confidence filtering │  │                                     │
       │                      │  │ ┌─ Standard Discovery ───────────┐ │
       │ Single AIFlow LLM    │  │ │ ~30 representative chunks      │ │
       │ call per cluster:    │  │ │ → 1 LLM call                   │ │
       │  Input: cluster      │  │ │ ~4.4s/cluster, 100% reliable   │ │
       │  metadata +          │  │ │ ~11 fields avg, broad coverage  │ │
       │  representative      │  │ └────────────────────────────────┘ │
       │  doc text            │  │                                     │
       │  Output: field_name, │  │ ┌─ RLM Discovery (DSPy CodeAct) ┐ │
       │  field_type,         │  │ │ All chunks via sandboxed REPL  │ │
       │  definition,         │  │ │ LLM writes Python → regex +   │ │
       │  examples,           │  │ │ strategic sampling → 3-6       │ │
       │  usefulness_score    │  │ │ batched sub-LLM calls →        │ │
       │                      │  │ │ consensus voting               │ │
       │ ~11 fields/cluster   │  │ │ ~36.5s/cluster, 95% reliable  │ │
       │ ~4.4s/cluster        │  │ │ (fallback → Standard on fail)  │ │
       │ 100% reliable        │  │ │ ~10% overlap with Standard     │ │
       │                      │  │ └────────────────────────────────┘ │
       │ Best for: fast       │  │                                     │
       │ iteration, low       │  │ ┌─ 4-Tier Merge ────────────────┐ │
       │ latency budgets,     │  │ │ T1: Exact name → "both"       │ │
       │ proven baseline      │  │ │ T2: Fuzzy substring → "fuzzy" │ │
       │                      │  │ │ T3: Standard-unique fields     │ │
       └──────────┬───────────┘  │ │ T4: RLM-unique fields         │ │
                  │              │ │ Capped at max_fields (def 15)  │ │
                  │              │ └────────────────────────────────┘ │
                  │              │                                     │
                  │              │ Hybrid results (CUAD-510 validated):│
                  │              │  13.4 fields/cluster (+22%)         │
                  │              │  0.751 avg confidence (highest)     │
                  │              │  0% failure rate                    │
                  │              │  ~$0.45 per full corpus run         │
                  │              │                                     │
                  │              │ Best for: production, max coverage, │
                  │              │ accounts needing deep field catalog  │
                  │              └───────────────┬─────────────────────┘
                  │                              │
    ┌─────────────▼──────────────────────────────▼────────────────────┐
    │ OUTPUT (identical schema regardless of backend)                  │
    │                                                                  │
    │  field_name, field_type, definition, examples,                   │
    │  usefulness_score, source_attribution                            │
    │  Types: text (39%), entity_name (14%), boolean (10%),            │
    │         date, duration, enum, monetary_amount                    │
    │                                                                  │
    │  Cluster-scoped definitions solve the core problem: field        │
    │  validity varies by contract type [^max-eval-observations].      │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ STAGE 4: FIELD DEDUPLICATION                                        │
│                                                                     │
│  Problem: fields like "effective_date" appear in many clusters      │
│       │                                                             │
│       ├─→ Embed all (field name + definition) pairs                │
│       ├─→ Cosine similarity merge at 0.93 threshold                │
│       ├─→ Identify Universal fields (appear in 3+ clusters)        │
│       └─→ Target: reduce raw fields to ~60-70% unique              │
│                                                                     │
│  69 Universal fields identified in eval [^aaditya-eval-dedup]       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                     ┌─────────▼─────────┐
                     │  FEATURE FLAG      │
                     │  extraction_backend│
                     └────┬─────────┬────┘
                          │         │
          ┌───────────────▼───┐ ┌───▼────────────────────────────────┐
          │ "sequential"      │ │ "rag" (default)                    │
          │ → Pipeline 3      │ │                                    │
          │ (see below)       │ │ STAGE 5: RETRIEVAL & EXTRACTION    │
          └───────────────────┘ │                                    │
                                │  Sample: Cochran formula            │
                                │  (z²·p·(1-p)/e²), capped 10 docs  │
                                │                                    │
                                │  For each (document, field) pair:  │
                                │                                    │
                                │  Round 1: Hybrid retrieval         │
                                │   ├─ Semantic: embed field def     │
                                │   │   → pgvector top-k chunks      │
                                │   ├─ Lexical: pg_trgm + tsquery   │
                                │   └─ Merge: RRF, top 3 chunks     │
                                │                                    │
                                │  Round 2+: Agentic retry (max 3)   │
                                │   — +3% F1 from eval               │
                                │   ├─ If "Insufficient Context":    │
                                │   │   LLM search terms → re-query  │
                                │   └─ Cross-doc: NL Index lookup    │
                                │       → heading-targeted retrieval  │
                                │                                    │
                                │  Round N: Reviewer agent (cheap)   │
                                │   └─ Accept/reject + re-extract    │
                                │                                    │
                                │  LLM Extraction (per doc):         │
                                │   ├─ Input: field name/def/type    │
                                │   │   + examples + snippets        │
                                │   ├─ Output: {value, evidence,     │
                                │   │           is_default}           │
                                │   ├─ Model: GPT-5.4 for complex   │
                                │   │   fields (renewal, termination)│
                                │   │   GPT-5.4-nano for simple      │
                                │   │   fields (dates, parties)      │
                                │   └─ Normalize: DATE→YYYY/MM/DD,  │
                                │       NUMBER preserve symbols      │
                                │                                    │
                                │  Write: AIDB (AddCluster →         │
                                │   AddClusterFieldSuggestions →      │
                                │   AddDocumentFieldValues in batches)│
                                │                                    │
                                │  User surface: GetFieldSuggestions  │
                                │   API → accept/retire feedback     │
                                │                                    │
                                │  NOTE: When extraction_backend=     │
                                │  "sequential", this stage is        │
                                │  replaced by Pipeline 3's Judge LLM │
                                │  compilation. Output schema is      │
                                │  identical — downstream consumers   │
                                │  see no difference.                 │
                                └───────────────┬────────────────────┘
```

---

## Pipeline 2: Clause-Level Clustering (followup)

Once the field discovery pipeline is shipping, clause-level clustering slots in as an upgrade to Stage 2. It replaces document-level clustering with clause-level clustering, which feeds into the **same downstream stages** (3–5) unchanged.

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2a: CLAUSE-TYPE CLUSTERING (EVoC)                            │
│                                                                     │
│  Input: clause chunks + clause embeddings from Stage 1              │
│         (structural split already done at ingest time)              │
│       │                                                             │
│       ├─→ EVoC + 30-trial Optuna (same as Pipeline 1)             │
│       ├─→ LLM labeling of each clause type                         │
│       ├─→ Near-duplicate merge at 0.96 cosine threshold            │
│       └─→ Outlier assignment to nearest type                        │
│                                                                     │
│  Output: ~79 clause types + 10 document domains (dual view)        │
│                                                                     │
│  Why: 6.7x field reusability (74.9 ext/field vs 11.1)             │
│  Fields map to real contractual concepts instead of topic blobs     │
│  Minimal field duplication (7 vs 56 duplicate names)                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                  Feeds into same Stages 3–5 from Pipeline 1
```

**What clause-level clustering improves** (from Aaditya's eval):
- Fields per cluster map to actual contractual concepts (e.g., "Independent Contractor", "Confidentiality Obligations") instead of document-type blobs
- 74.9 extractions/field (vs 11.1 at document level) — fields are reusable across documents
- Only 7 field name duplicates across clusters (vs 56 at document level)
- 79 clause types provide finer semantic granularity than 13 (HDBSCAN) or 10 (EVoC domains)

---

## Pipeline 3: Sequential Chunk Extraction (alternate extraction backend)

Max's latest experiment proposes eliminating the RAG retrieval step entirely. Instead of "find the right chunks for each field," process ALL chunks sequentially, extract ALL fields from each chunk, carry forward knowledge across chunks, then compile with a Judge LLM.

**Motivation:** Experian extracts ~200 fields per run (min: 66, max: 313). RAG-based extraction requires a search index per account and scales as O(fields × retrieval_calls). Sequential extraction scales as O(chunks) regardless of field count — the field catalog is injected once into the extractor prompt.

**Approach:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ SEQUENTIAL CHUNK EXTRACTION (replaces Stage 5)                     │
│                                                                     │
│  Input: chunks from Stage 1 + field catalog from Stage 4            │
│                                                                     │
│  Step 1: Split document into ~4000 tok chunks                       │
│          (larger than embedding chunks — optimized for extraction,  │
│           not retrieval. Best chunk size needs more experimentation) │
│                                                                     │
│  Step 2: Extractor LLM processes each chunk sequentially            │
│    ├─ Input: chunk text + full field catalog + carryforward context │
│    ├─ Output per chunk:                                             │
│    │   • extracted {field, value, evidence} for fields found here   │
│    │   • carryforward notes for future chunks                       │
│    │     (e.g., "effective date = latter of two signatures")        │
│    │   • field reasoning tools for definition-aware extraction       │
│    └─ Model: 5.4-mini or 3.1-flash-lite (small model works —       │
│         context rot is mitigated by small chunk windows)             │
│                                                                     │
│  Step 3: Judge LLM compiles all chunk-level extractions             │
│    ├─ Input: all per-chunk extractions + evidences + field defs     │
│    ├─ Resolves conflicts (same field, different values across       │
│    │   chunks — pick best evidence)                                 │
│    ├─ Validates formatting against field type definitions            │
│    └─ Output: final {field, value, evidence, is_default} per field  │
│                                                                     │
│  Hooks into summary tokens: document summary can be a byproduct    │
│  of the sequential pass (same tokens, one extra output field)       │
└─────────────────────────────────────────────────────────────────────┘
```

**Knowledge carryforward** is the key innovation: the extractor writes notes for future chunks ("The effective date is equivalent to the latter of two signatures," "Party A = Acme Corp, established in preamble"). This lets later chunks resolve cross-references without re-reading earlier text. Effectively gives the model multiple reasoning passes over the same document.

**When to use which extraction backend:**

| Dimension | RAG (Stage 5) | Sequential (Pipeline 3) |
|---|---|---|
| Field count | <50 fields | 50–300+ fields |
| Document length | Any | Long docs benefit most (less context rot) |
| Infra requirement | pgvector + search index per account | None — just chunks in order |
| Scaling cost | O(fields × retrieval) | O(chunks) — fixed regardless of field count |
| Token efficiency | ~1k tok/field (targeted snippets) | More output tokens (exhaustive per chunk) |
| Search index needed | Yes | No |
| Summary synergy | Separate process | Byproduct of same pass |
| Maturity | Proven (agentic retry +3% F1) | Early — promising results on 5.4-mini |

**Feature flag:** `extraction_backend: "rag" | "sequential"` at the account level. Accounts with <50 fields use RAG (proven, lower token cost). Accounts with 50+ fields use sequential (better economics, no index). Both read the same field catalog from Stage 4.

**Known issues to solve:**
- Judge LLM over-rejects valid extractions (requires high confidence when asked to judge) — needs calibration of acceptance threshold or softer judge prompt [^max-eval-seq-judge]
- Uses more output tokens than ideal per chunk — the exhaustive extraction is useful but needs output budget control
- Optimal extraction chunk size (~4000 tok) needs tuning — different from embedding chunk size (2000 tok)
- Tested on 5.4-mini only; 5.4 kept 429'ing. 3.1-flash-lite is a worthwhile experiment for cost [^max-eval-seq-ceiling]

**F1 ceiling:** Max estimates ~85% based on label interpretation ambiguity — SX set failures increasingly look like annotation disagreements rather than extraction errors [^max-eval-seq-ceiling].

---

## Unified Pipeline: All Pipelines Coexisting

All three pipelines share Stage 1 (ingest & embed) and Stages 3–4 (field discovery & dedup). The extraction backend (Stage 5 vs Pipeline 3) is feature-flagged per account.

```
┌─────────────────────────────────────────────────────────────────────┐
│ SHARED STAGE 1: INGEST & EMBED (run once)                          │
│                                                                     │
│  Documents → Embedding chunks (2000 tok) → nomic embeddings        │
│  Documents → Clause chunks (structural split) → nomic embeddings   │
│  Documents → Extraction chunks (4000 tok, no embeddings)            │
│  Documents → NL Index (TOC/headings)                                │
│  All stored in pgvector, tagged by chunk_type                       │
│                                                                     │
│  All pipelines read from the same stored data.                      │
└──────────────┬──────────────────────────────────┬───────────────────┘
               │                                  │
    ┌──────────▼──────────┐            ┌──────────▼──────────┐
    │ PIPELINE 1           │            │ PIPELINE 2           │
    │ Document Clustering  │            │ Clause Clustering    │
    │                      │            │                      │
    │ mean-pool chunks     │            │ structural split     │
    │   → 1 vec/doc        │            │   → 1 vec/clause     │
    │ EVoC → ~10 domains   │            │ EVoC → ~79 types     │
    │                      │            │   + ~10 domains      │
    └──────────┬───────────┘            └──────────┬───────────┘
               │                                   │
    ┌──────────▼───────────────────────────────────▼──────────┐
    │ SHARED STAGES 3–4                                       │
    │                                                         │
    │  Stage 3: Field Discovery                               │
    │    Feature flag: discovery_backend = "standard"|"hybrid" │
    │    "standard" → single AIFlow LLM call (legacy)         │
    │    "hybrid"   → Standard + RLM + 4-tier merge (default) │
    │  Stage 4: Field Deduplication (cross-cluster + cross-   │
    │           pipeline merge of universal fields)            │
    └──────────────────────────┬──────────────────────────────┘
                               │
                     ┌─────────▼─────────┐
                     │  FEATURE FLAG      │
                     │  extraction_backend│
                     └────┬─────────┬────┘
                          │         │
           ┌──────────────▼──┐  ┌───▼──────────────────┐
           │ "rag" (default)  │  │ "sequential"          │
           │                  │  │                       │
           │ Stage 5:         │  │ Pipeline 3:           │
           │ Retrieval &      │  │ Sequential Chunk      │
           │ Extraction       │  │ Extraction            │
           │                  │  │                       │
           │ Best for: <50    │  │ Extractor + Carry-    │
           │ fields, proven   │  │ forward → Judge LLM   │
           │                  │  │                       │
           └────────┬─────────┘  │ Best for: 50-300+     │
           │                  │  │ fields, no search idx │
           └────────┬─────────┘  └───────────┬──────────┘
                    │                        │
    ┌───────────────▼────────────────────────▼─────────────┐
    │ OUTPUT                                               │
    │                                                      │
    │  View 1: Document Domains (~10 groups)               │
    │  View 2: Clause Types (~79 types)  [if Pipeline 2]   │
    │  Extractions: same schema regardless of backend      │
    │  Bonus: document summary (free with sequential path) │
    └──────────────────────────────────────────────────────┘
```

**How dedup works across pipelines:** Stage 4 runs once over the union of fields from both clustering views. Universal fields ("effective_date", "governing_law") are merged. View-specific fields are preserved with a `source_view` tag. Result: one deduplicated field catalog, two navigation views.

**Ordering:** Pipeline 1 ships first with RAG extraction (default). Pipeline 3 (sequential) ships as a feature-flagged alternative once the Judge calibration is solved. Pipeline 2 (clause clustering) plugs in independently as a Stage 2 upgrade. No changes to downstream stages required for any combination.

---

## F1 Improvement Levers

Beyond the core pipeline stages, these improvements target specific failure modes and quality gaps. Ranked by expected F1 impact.

### High Impact

**1. Outlier Recovery Strategy** (targets 26.2% outlier rate)

The current proposal assigns outliers to the nearest cluster, which forces mismatched field definitions onto ~26% of documents. Two better approaches:

- **Iterative re-clustering**: Re-run EVoC on the outlier pool at lower `base_min_cluster_size`. Creates small "rare contract type" clusters (e.g., 3-doc equipment leases) instead of force-fitting them into the nearest domain. Fields discovered from these micro-clusters actually match the documents.
- **Soft assignment**: Assign each outlier to top-2 nearest clusters. Run field discovery against both. Keep whichever produces higher-confidence extractions per field. More LLM cost but no documents get wrong definitions.

Expected impact: +1-2% F1 by avoiding extraction errors on 26% of the corpus.

**2. Field Definition Calibration Loop** (targets definition ambiguity)

Stage 3 generates field definitions once from representative docs and never revises them. Add a calibration pass after initial discovery:

1. Extract the field from 3-5 docs using the initial definition
2. Check if extracted values are type-consistent (e.g., all dates, not a mix of dates and paragraphs)
3. If inconsistent: refine the definition with counter-examples, re-extract
4. If still inconsistent: flag as ambiguous, split into two fields or downgrade confidence

Cost: ~5 extra LLM calls per field. Directly tightens extraction targets before running on the full corpus.

Expected impact: +1-2% F1 by catching ambiguous definitions ("Effective Date" extracting both dates and full paragraphs).

**3. Negative Examples in Extraction Prompts** (targets confusion errors)

Max's eval identifies incidental party mentions and ContractType overlap as top failure cases [^max-eval-failure-cases]. Both are confusion errors — the LLM finds a plausible-looking value that isn't the target.

Add to extraction prompts: `"This field is NOT: [counter-examples]"` sourced from:
- The calibration loop (values that looked right but were wrong)
- Cluster metadata (adjacent fields that are commonly confused)
- User retire feedback (fields users rejected)

Expected impact: +0.5-1% F1 on high-confusion fields (parties, dates, contract types).

**4. Accept/Retire Feedback Loop** (targets continuous improvement)

The proposal has `GetFieldSuggestions → accept/retire` but that signal is never fed back. Close the loop:

- **Accepted fields**: Promote definition confidence. Use accepted definitions as-is in future runs for the same cluster. Over time, the field catalog converges to human-validated definitions.
- **Retired fields**: Suppress similar fields (cosine > 0.90 to retired definition) in future discovery runs. Prevents re-suggesting fields users don't want.
- **Accepted extractions**: Accumulate as few-shot examples for that field. After 5+ accepted examples, add the best 2-3 to the extraction prompt as demonstrations.

Expected impact: +1-3% F1 over time (cheapest gain — users provide labeled data for free). Compounds with every batch of user feedback.

### Medium Impact

**5. Field-Type-Specific Retrieval Strategies** (targets snippet precision)

One retrieval path for all 7 field types is suboptimal. Different types cluster in different document locations:

| Field Type | Where It Lives | Retrieval Strategy |
|---|---|---|
| `date` | Signature blocks, headers, first/last pages | Prioritize structural chunks (preamble, signature) |
| `entity_name` | Preambles, "BETWEEN" clauses | Prioritize first 3 chunks + "parties" heading |
| `boolean` | Existence detection, not value extraction | Lower k, check for presence/absence keywords |
| `monetary_amount` | Payment/fee sections | NL Index → heading-targeted retrieval |
| `text` / `enum` | Scattered throughout | Standard hybrid retrieval (default path) |
| `duration` | Renewal, termination, notice sections | NL Index → section-targeted retrieval |

Implementation: Small routing table (field_type → retrieval config) in Stage 5. No architecture change.

Expected impact: +0.5-1% F1 by improving snippet precision for structured field types.

**6. Confidence-Gated Abstention** (targets precision over recall)

When the LLM isn't confident, it should return "not found" rather than hallucinate a value. Max's eval notes absence validation is the hardest failure case [^max-eval-failure-cases].

- Add a confidence threshold to extraction output (e.g., `confidence: float 0-1`)
- Below threshold → return `{value: null, reason: "low_confidence"}` instead of a guess
- Count abstentions separately from errors in metrics
- Trade a small recall drop for a larger precision gain → net F1 improvement

Expected impact: +0.5-1% F1 by eliminating hallucinated values on ambiguous fields.

**7. Clause-Aware Chunking** (targets embedding quality)

The 2000-tok / 100-overlap fixed window cuts across clause boundaries. When a chunk contains the end of one clause and the start of another, the embedding represents mixed semantic content → noisier clustering and retrieval.

Fix: Split at structural boundaries first (paragraph breaks, section markers, numbered clauses), then merge adjacent segments up to the token budget. This produces chunks that align with contractual concepts.

Expected impact: +0.5% on clustering quality score, +0.5% on retrieval F1. Compounds — better embeddings improve both clustering and snippet selection.

### Longer Horizon

**8. Contrastive Fine-Tune of Nomic** (targets domain-specific retrieval)

Train on `(field_definition, relevant_chunk)` positive pairs from accepted extractions. Creates a domain-adapted embedding that directly optimizes retrieval for contract field extraction.

Requires: 1,000+ accepted extraction pairs (accumulated over weeks of production use). Not viable on day one but becomes the highest-ceiling improvement once data exists.

Expected impact: +2-4% retrieval F1 (based on typical domain adaptation gains on legal text).

**9. Cross-Document Consistency Check** (targets extraction outliers)

After extraction, flag fields where value distributions are inconsistent within a cluster:
- If "Governing Law" returns a state name in 80% of docs but a full paragraph in 20%, something is wrong for that 20%
- Batch-flag inconsistent extractions for re-extraction with tighter prompts or manual review
- Also catches OCR/parsing errors that produce garbage values

Expected impact: +0.5-1% F1 by catching and correcting extraction outliers in batch.

**10. Hybrid Sequential + RAG Extraction** (targets high-field accounts)

For accounts with 50+ fields: run the sequential chunk extractor (Pipeline 3) as a first pass to get exhaustive extractions cheaply. Then run targeted RAG retrieval (Stage 5) only on fields where the Judge LLM flagged low confidence or conflicting evidence. Combines the breadth of sequential with the precision of targeted retrieval.

Expected impact: Best-of-both-worlds extraction quality at scale. The sequential pass catches fields that RAG would miss (no index required), while the RAG pass rescues fields where the Judge was uncertain.

---

## Key Architecture Decisions

| Decision | Rationale | Source |
|---|---|---|
| **EVoC only (no HDBSCAN)** | 0.920 quality, 38% faster, 4 params, no UMAP, cosine-native | [^aaditya-eval-evoc] |
| **Decouple from schematizer** | Enable independent iteration on embeddings, retrieval, prompts | [^max-eval-decouple] |
| **Upgrade embeddings to nomic** | +6% retrieval F1, independently validated by both evals | [^max-eval-embeddings] [^aaditya-eval-nomic] |
| **Agentic snippet selection** | +3% F1 on scattered-context fields (dates, terms) | [^max-eval-agentic] |
| **Cluster-scoped field definitions** | Field validity varies by contract type | [^max-eval-observations] |
| **Field deduplication** | 56 duplicate names at doc-level; 69 universal fields identified | [^aaditya-eval-dedup] |
| **Tiered LLM routing** | Thinking models help renewal/termination; nano for dates | [^max-eval-sx-results] |
| **Clause pipeline as followup** | Document-level EVoC ships field discovery now; clause-level is an upgrade to Stage 2 only | [^aaditya-eval-hybrid-rec] |
| **Sequential extraction as flag** | O(chunks) vs O(fields×retrieval); better economics at 50+ fields; no search index | [^max-eval-seq-approach] |
| **Knowledge carryforward** | Cross-chunk references ("effective date = latter of two signatures") without re-reading | [^max-eval-seq-approach] |

---

## Pipeline Comparison (from Aaditya's eval)

| Metric | HDBSCAN Macro-Micro | HDBSCAN Clause-Type | **EVoC Hybrid** |
|---|---|---|---|
| Algorithm | BERTopic/HDBSCAN + Optuna | Flat HDBSCAN | **EVoC + Optuna** |
| Groups | 7 domains → 80 clusters | 13 clause types | **10 domains + 79 clause types** |
| Total fields | 918 | 158 | **944** |
| Extractions/field | 11.1 | 74.9 | 11.2 |
| Avg quality score | 0.908 | 0.883 | **0.920** |
| Total time | 33.4 min | 22.2 min | **20.7 min** |
| Clustering time | 973s | 468s | **~160s** |
| Outlier rate | **5.2%** | 18.3% | 26.2% |
| Field duplication | 56 names in 2+ clusters | 7 names in 2+ clusters | TBD (dedup stage) |

---

## What Changes vs Current Production

1. **Clustering algorithm**: UMAP + HDBSCAN → EVoC (no dimensionality reduction, cosine-native)
2. **Embedding model**: MiniLM-v3-384 → nomic-embed-text-v1.5 (768-dim, Apache licensed)
3. **Optimization**: 6 HDBSCAN hyperparams → 4 EVoC params (base_min_cluster_size, noise_level, merge threshold, layer selection)
4. **Retrieval**: Single HyDE vector search → hybrid (semantic + lexical) with agentic retry
5. **Document index**: None → one-time NL index generation (TOC/heading structure)
6. **Post-clustering**: None → field deduplication (0.93 cosine merge)
7. **Extraction routing**: Single model → tiered by field complexity
8. **Extraction backend**: Single RAG path → feature-flagged RAG or sequential chunk extraction
9. **Followup**: Document-level clustering → clause-level clustering (Pipeline 2)

---

## Implementation Priority

### Pipeline 1 (Field Discovery — now)

| Priority | Task | F1 Lever |
|---|---|---|
| **P0** | Replace HDBSCAN with EVoC in clustering stage | Core |
| **P0** | Reduce EVoC outlier rate to <20% — iterative re-clustering or soft assignment | #1 Outlier Recovery |
| **P0** | Integrate nomic-embed-text-v1.5 embeddings | Core |
| **P0** | Clause-aware chunking (structural boundary splitting) | #7 |
| **P1** | Field definition calibration loop (extract 3-5 docs → refine) | #2 |
| **P1** | Negative examples in extraction prompts | #3 |
| **P1** | Field deduplication stage (cosine merge + universal field identification) | Core |
| **P1** | Agentic snippet selection with hybrid retrieval | Core |
| **P1** | Field-type-specific retrieval strategies | #5 |
| **P1** | Tiered LLM routing for extraction | Core |
| **P1** | Confidence-gated abstention (null over hallucination) | #6 |
| **P2** | Accept/retire feedback loop (promote, suppress, few-shot) | #4 |
| **P2** | Cross-document consistency check | #9 |
| **P2** | Incremental clustering mode (new docs without full re-cluster) | — |
| **P3** | Contrastive fine-tune of nomic on accepted extractions | #8 |

### Pipeline 2 (Clause Clustering — followup)

| Priority | Task | F1 Lever |
|---|---|---|
| **P0** | Clause extraction (structural document split) | Core |
| **P0** | Replace document-level EVoC with clause-level EVoC in Stage 2 | Core |
| **P1** | Dual-view output (clause types + document domains) | Core |
| **P1** | Per-type micro clustering for long-tail field coverage | Core |
| **P1** | Cross-pipeline field deduplication (unified catalog, two views) | Core |

### Pipeline 3 (Sequential Chunk Extraction — feature flag)

| Priority | Task | F1 Lever |
|---|---|---|
| **P0** | Calibrate Judge LLM acceptance threshold (currently over-rejects) | Core |
| **P0** | Implement knowledge carryforward between chunks | Core |
| **P0** | Feature flag `extraction_backend: rag \| sequential` per account | Core |
| **P1** | Tune extraction chunk size (~4000 tok, separate from embedding chunks) | Core |
| **P1** | Output budget control (exhaustive extraction produces excess tokens) | Core |
| **P1** | Document summary as byproduct of sequential pass | Bonus |
| **P2** | Test 3.1-flash-lite as extractor (cost optimization) | — |
| **P2** | Hybrid mode: sequential for first pass, RAG for targeted re-extraction | #10 |

---

## F1 Target Trajectory

- Current production (ur_multilingual + GPT-4o-mini FT): **82.7%**
- + Embedding upgrade (nomic) + clause-aware chunking: **~84-85%**
- + Hybrid retrieval + field-type routing: **~85-86%**
- + Outlier recovery + field calibration loop: **~87-88%**
- + Agentic snippet selection + negative examples + abstention: **~88-89%**
- + Accept/retire feedback loop (compounds over time): **~89-90%**
- + Sequential chunk extraction for high-field accounts (Pipeline 3): **~89-90%** (parallel path)
- + Clause-level clustering (Pipeline 2 — field reusability): **~90%+**
- + Contrastive fine-tune of nomic (longer horizon): **~91-92%**
- Dataset ceiling (annotation ambiguity): **~85-90%** [^max-eval-observations] [^max-eval-seq-ceiling]

---

## Codebase Impact

The existing `dynamic_scaler.py` needs a new profile for EVoC's 4 params:
- `base_min_cluster_size` (Optuna found optimal=6)
- `noise_level` (Optuna found optimal=0.30)
- `merge_threshold` (0.96)
- Layer selection scoring weights

When Pipeline 2 (clause-level) ships, the scaler also needs a `clustering_unit` parameter since clause-level input has ~20-40x more items than document count (3,159 chunks / 2,002 clauses from 98 docs).

Pipeline 3 (sequential extraction) requires a separate chunking pass at ~4000 tokens (vs 2000 for embeddings). These extraction chunks should be stored alongside embedding chunks in the same pgvector schema but tagged with `chunk_type: "extraction"`. The carryforward context accumulator and Judge LLM are new components with no existing codebase equivalent — Max's Docusheets prototype (NodeJS + React + Python shim) has the architecture but needs a proper rewrite [^max-eval-seq-docusheets].

---

## Appendix A: Sources from Max's Eval

All `[^max-eval-*]` footnotes reference "Clustering Extractions Pipeline Evals" (Max's eval).

[^max-eval-decouple]: **Page 1, para 1** — *"Our initial clustering extractions pipeline existed as part of the field schematizer, in an intertangled web of dependencies. This made it difficult to iterate on large changes, experiment with new embedding models, etc."*

[^max-eval-sx-results]: **Page 1, SX Eval Results table** — GPT-5.4 reaches 0.917 F1 on RenewalOrNonRenewalNoticePeriod vs GPT-4o-mini FT at 0.69. Thinking models dramatically help on complex fields, while simpler fields show minimal model delta.

[^max-eval-observations]: **Page 2, "General Feeling"** — *"The standard fields are our best attempt to box in and categorize each contract with a single comparable value. But the definitions and validity of each vary substantially by type of contract. Clustering related fields gives us fields that should be matching the set of valid extractions."*

[^max-eval-context-rot]: **Page 3, Baseline 1** — Naive full-doc extraction: 81.5% avg F1, with *"higher error rates for longer documents... Context rot is likely the limiting factor."*

[^max-eval-agentic]: **Page 6-7, "Multi Step Snippet Selection"** — *"Lacking the right context is the primary reason for extractions failure."* Agentic retry yielded **+3% F1 (71→74%)** on gpt-5.4-mini.

[^max-eval-nl-index]: **Page 7, "Document NL Index"** — *"If a human were tasked with finding the 'Recycling Fee' in a huge 172 page franchise agreement, and Ctrl-F failed, the human may try find either a TOC, or skim headings."*

[^max-eval-embeddings]: **Page 10, corrected results** — Retrieval+extraction F1: pplx-q4 76.5%, nomic 76.3%, harrier 71.4%, MiniLM 70.0%.

[^max-eval-failure-cases]: **Page 2, "Main Failure cases"** — ContractType overlap, incidental party mentions, "latter of two signatures" dates, vague notice periods, absence validation hardest.

[^max-eval-overfitting]: **Page 5, Worktree 1** — Agentic coding overfits aggressively (55%→95% on 100 docs, drops to 70-80% on unseen). Argues config-driven extraction over hardcoded rules.

---

## Appendix B: Sources from Max's Sequential Extraction Experiment

All `[^max-eval-seq-*]` footnotes reference Max's sequential chunk extraction experiment (April 2026). Tested on SX eval set with GPT-5.4-mini. Report at `https://20.80.172.76/report.html`. Prototype codebase: Docusheets (NodeJS + React + Python service-cluster).

[^max-eval-seq-approach]: **Experiment motivation** — Experian extracts ~200 fields per run (min: 66, max: 313). RAG needs search indexes (complex infra) and doesn't scale to 200+ extractions at ~1k tokens each. Sequential processing: split doc into ~4000 tok chunks, run LLM extractor per chunk with knowledge carryforward, Judge LLM compiles final results.

[^max-eval-seq-carryforward]: **Knowledge carryforward** — Extractor writes future-helpful context between chunks (e.g., "The effective date is equivalent to the latter of two signatures"). Gives model tools to reason about definitions while maintaining minimal context. Effectively provides multiple reasoning passes over the same content.

[^max-eval-seq-judge]: **Judge LLM issue** — *"When a model is asked to be a JUDGE, it suddenly requires very high confidence, and was eliminating too many valid extractions."* The RLM-like judge approach needs calibration — over-rejection is the primary failure mode. Softer acceptance thresholds or restructured judge prompts needed.

[^max-eval-seq-ceiling]: **F1 ceiling estimate** — *"The more I look at the fail cases in SX set, I'm starting to think some of these labels are just not fully correct. The ceiling is probably ~85% based on interpretation."* Tested on 5.4-mini (5.4 kept 429'ing). 3.1-flash-lite also a worthwhile experiment.

[^max-eval-seq-benefits]: **Benefits** — (1) Hook into same tokens used for summary generation. (2) Less affected by context rot — smaller model performs well as extractor on small windows. (3) Multiple reasoning passes over same context. (4) Exhaustive extraction list — richer, more token-efficient analysis than 100k+ token input docs.

[^max-eval-seq-tokens]: **Token concern** — *"Overall my feeling is it uses too many tokens, is too slow."* Current implementation produces more output tokens than ideal per chunk. Sequential pass is thorough but needs output budget control.

[^max-eval-seq-docusheets]: **Prototype** — Docusheets: NodeJS backend + React frontend + Python service-cluster (shim for hackathon clustering logic). PostgreSQL-backed with pgvector, pg_trgm, optional pg_textsearch BM25. *"It's super copilot gen, please consider taking just the ideas and architecting it properly. It's bloated with a bunch of rubbish & experiments."*

---

## Appendix C: Sources from Aaditya's Eval

All `[^aaditya-eval-*]` footnotes reference "Chunking Strategies / Algorithm Strategies / E2E Pipeline" (Aaditya's eval). Appendix renumbered — was Appendix B prior to sequential extraction experiment. Dataset: 98 public legal agreements, 3,159 chunks, ~2,002 valid clauses. Hardware: 2× A100 80GB, gpt-4o-mini. Embeddings: nomic-embed-text-v1.5 768-dim.

[^aaditya-eval-metrics]: **Chunking, Section 2 "Head-to-Head Metrics"** — Macro-Micro: 918 fields, 11.1 ext/field, 56 duplicate names. Clause-Type: 158 fields, 74.9 ext/field, 7 duplicate names. Clause-type fields are 6.7x more reusable.

[^aaditya-eval-evoc]: **Algorithm, "Key Takeaway: EVoC Hybrid"** — EVoC Hybrid with Optuna: 79 clause types, 10 document domains, 944 fields, 0.920 avg quality (highest), 20.7 min total (38% faster than HDBSCAN M-M). Clustering alone ~160s, 6x faster than HDBSCAN M-M's 973s.

[^aaditya-eval-nomic]: **Chunking, Section header** — nomic-embed-text-v1.5 (768-dim, CUDA) used as embedding model across all pipeline variants. Independent validation of Max's retrieval eval results.

[^aaditya-eval-dedup]: **Chunking, Section 2** — Macro-Micro: 56 field names appear in 2+ clusters. Clause-Type: only 7 duplicates. **E2E, Section 6** — EVoC Hybrid identified 69 Universal fields. *"P1: Cross-type field dedup (reduce 944 to ~600 unique fields)."*

[^aaditya-eval-evoc-params]: **Algorithm, "What Changed: EVoC V1 to V2"** — Key tuning: layer selection changed to scoring-based, base_min_cluster_size=6 (was 40), noise_level=0.30 (was 0.5), merge threshold raised from 0.93 to 0.96.

[^aaditya-eval-evoc-cons]: **Algorithm, "Pros and Cons: EVoC Hybrid"** — Cons: 26.2% outlier rate, early beta (v0.3.1), no c-TF-IDF keyword extraction.

[^aaditya-eval-clause-quality]: **Chunking, Section 3 "Cluster Quality"** — Macro-Micro: 0.908. Clause-Type: 0.883. EVoC Hybrid: 0.920 (highest). EVoC discovers 6x more clause types (79 vs 13).

[^aaditya-eval-timing]: **Algorithm, "Speed Comparison"** — HDBSCAN M-M: 33.4 min. EVoC Hybrid: 20.7 min. EVoC M-M: 11.8 min. Field discovery dominates EVoC time at 977s.

[^aaditya-eval-field-types]: **Chunking, Section 4 "Field Type Distribution"** — Top types: text (39.2%), entity_name (13.9%), boolean (10.1%).

[^aaditya-eval-p0]: **E2E, Section 6 "Implementation Priority"** — P0: Reduce outlier rate (<20%).

[^aaditya-eval-hybrid-rec]: **Chunking, Section 7 "Recommendation"** — *"Best of both: Hybrid approach (recommended). Steps: Clause extraction → Clause-type clustering (for recs) → Per-type micro clustering (for granularity) → Field discovery on both levels."*
