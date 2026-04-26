# EVoC Extraction Pipeline — Architecture Analysis & Comparison

## 1. Benchmark Results: DSPy v2 vs Baseline (5 clusters)

### Aggregate Summary

| Metric | DSPy v2 | OG rlm_v2 | OG dspy_rlm | OG std_gpt | OG rlm_lb |
|---|---|---|---|---|---|
| **Avg fields/cluster** | 15.0 | 15.0 | 8.6 | 12.0 | 8.0 |
| **Avg fill rate** | 0.312 | 0.323 | 0.288 | 0.308 | 0.413 |
| **Avg verbatim rate** | 0.726 | 0.773 | 0.926 | 0.890 | 0.859 |
| **Avg grounded rate** | 0.736 | 0.787 | 0.974 | 0.897 | 0.942 |
| **Avg hallucination rate** | **0.264** | 0.213 | **0.026** | 0.072 | 0.058 |
| **Avg discovery time** | 55.1s | 91.4s | 2.2s | 3.8s | 2.9s |
| **Avg extraction time** | 10.4s | 8.3s | 5.6s | 7.6s | 6.7s |

### Per-Cluster Breakdown

| Cluster | DSPy v2 halluc | OG rlm_v2 halluc | OG std_gpt halluc | OG dspy_rlm halluc |
|---|---|---|---|---|
| Franchise agreement | **72.5%** | 32.0% | 10.7% | 0.0% |
| Offering Allocation | 14.3% | 15.2% | 4.9% | 10.5% |
| Website Co-Branding | 12.5% | 7.9% | 3.6% | 0.0% |
| Defined Terms | 12.2% | 13.0% | 0.0% | 0.0% |
| Stock option grants | 20.5% | 38.5% | 16.7% | 0.0% |

### Key Observation

**Excluding the Franchise outlier**, DSPy v2 averages **~15% hallucination** — competitive with OG rlm_v2's ~18.7%. The Franchise disaster is a field-design problem, not a pipeline problem.

---

## 2. Root Cause Analysis

### 2.1 The Franchise Disaster (72.5% hallucination)

**What happened**: DSPy's ChainOfThought discovered **meta-fields** instead of extractable content fields:

| DSPy v2 Fields (Franchise) | OG rlm_v2 Fields (Franchise) |
|---|---|
| `advertising_fee_section_present` (boolean) | `agreement_title` (text) |
| `training_section_present` (boolean) | `franchisor_name` (text) |
| `operations_manual_section_present` (boolean) | `franchise_renewal_right` (text) |
| `royalty_fee_section_present` (boolean) | `protected_territory_status` (text) |
| `computer_system_requirement` (string) | `franchise_term_reference_exhibit` (text) |

The DSPy signature says "discover FIELDS whose values vary across contracts" but ChainOfThought interpreted this as "document what sections exist" — producing boolean presence-checks that can't have verbatim spans.

**Root cause**: The `GroundedDiscoverySig` prompt lacks a hard constraint on field extractability. The OG rlm_v2 pipeline has inline validation that penalizes fields without span_text, but DSPy's ChainOfThought doesn't get that feedback loop.

**Fix**: Add an explicit constraint to the signature: *"Every field MUST have a span_text quote. Fields answerable only by yes/no or whose values cannot be copied verbatim from the text are FORBIDDEN."*

### 2.2 Overlap Merge Not Filtering

For Franchise: `explore_kept=115, explore_dropped=0, explore_ratio=1.00`. The overlap merge kept **every single exploratory field**. This means the cosine threshold (0.85) never triggered — exploratory meta-fields are semantically distant from grounded content fields, so they all pass.

**Fix**: Add a secondary filter: reject exploratory fields whose `span_text` is empty or whose `value_type` is `boolean`.

### 2.3 Fill Rate Stagnation

All methods show fill rates between 0.27–0.41. This means for most (field, chunk) pairs, the field doesn't apply to that chunk — which is expected since fields are cluster-level, not document-level. But **fill rate is the real bottleneck** for production value: if a field applies to only 30% of chunks, users need to process 3 chunks to find one value.

### 2.4 Discovery Quality Score vs Grounding

DSPy v2's internal `quality_score` is 0.9834 — it thinks its fields are excellent. But grounding is 0.264 hallucination. The quality metric (which scores diversity, specificity, coverage) is disconnected from the actual extraction outcome. **The metric optimizes for interesting fields, not extractable fields.**

---

## 3. What's Missing: Gap Analysis

### Gap 1: No Extraction-Aware Discovery

**Current flow**:
```
Discover fields (LLM) → Extract values (LLM) → Score grounding (string match)
```

The discovery step never gets feedback from extraction. Fields like `advertising_fee_section_present` score well on discovery metrics (specific, diverse) but fail at extraction. There is no closed loop.

**What's needed**: A "trial extraction" step between discovery and finalization. After discovering candidate fields, run extraction on 2-3 sample chunks and drop fields with <50% fill rate or >30% hallucination rate.

### Gap 2: No Cross-Document Field Deduplication

Each cluster's fields are discovered independently. But two clusters might both discover `agreement_date` or `party_name`. There's no global field registry to normalize across clusters.

**Impact**: When a user looks at extractions across document types, the same conceptual field might appear under 5 different names. This makes cross-cluster analytics impossible.

### Gap 3: No Field-Value Conflict Resolution

When the same field is extracted from multiple chunks of the same document, there's no mechanism to resolve conflicts. Chunk 1 might say `termination_notice_period = "30 days"` and chunk 5 might say `termination_notice_period = "60 days"`. The current pipeline stores both without flagging the contradiction.

### Gap 4: No Document-Level Context

The pipeline processes **clusters of chunks**, not documents. A chunk from document A and a chunk from document B get mixed into the same cluster batch. The LLM doesn't know which chunks come from the same document, which prevents:
- Carryforward context (what was learned from earlier chunks of this doc)
- Cross-reference resolution ("as defined in Section 2" → look at Section 2 chunk)
- Document-level field schemas (some fields apply to entire documents)

### Gap 5: No Retrieval Optimization

All methods process chunks in cluster order (random or embedding-sorted). There's no retrieval step that asks: "Given field X, which chunks are most likely to contain its value?" This means the pipeline wastes LLM calls on chunks that can't contribute.

### Gap 6: No Quality Feedback Loop into Prompts

MIPROv2 can optimize prompts, but only if we have a training signal. Currently the "quality" metric scores discovery aesthetics, not extraction accuracy. We need a metric that measures: "Did the discovered fields lead to high-grounding extractions?"

---

## 4. Architecture Comparison: Current vs Friend's Approaches

### 4.1 Our Architecture: Cluster-First Discovery

```
Documents → Chunk → Embed → Cluster → Discover Fields (per cluster) → Extract Values
```

**Philosophy**: Fields emerge from patterns across similar clauses. Group first, then discover.

**Strengths**:
- Discovers **novel fields** that no pre-existing schema covers
- Works on unseen contract types without any labeled data
- Cluster-level fields capture what makes a clause type unique
- Rolling state + latent briefing handles large clusters efficiently

**Weaknesses**:
- No document-level context during extraction
- No retrieval optimization (processes chunks linearly)
- No conflict resolution for multi-chunk values
- Discovery quality disconnected from extraction quality
- High hallucination when discovery produces non-extractable fields

---

### 4.2 Friend's Approach A: RAG-Based Extraction

```
Pre-existing field catalog
    ↓
Per-document: hybrid retrieval (pgvector + pg_trgm + RRF scoring)
    ↓
Cochran sampling picks relevant chunks per field
    ↓
Extractor LLM (GPT-4o for complex fields, GPT-4o-mini for simple)
    ↓
Agentic retry loop (max 3 rounds if value missing)
    ↓
Cross-doc NL index lookup for context enrichment
    ↓
Reviewer agent verifies extractions
    ↓
AIDB write path
```

**Key Design Decisions**:

| Feature | Description | Why It Matters |
|---|---|---|
| **Cochran sampling** | Statistical sampling to pick top-k chunks per field with confidence bounds | Avoids processing all chunks — only retrieves chunks likely to contain the field value |
| **Hybrid retrieval** | Combines dense embeddings (pgvector) with trigram text matching (pg_trgm) via Reciprocal Rank Fusion | Catches both semantic and lexical matches — "termination" and "early exit" both retrieve |
| **Tiered LLM routing** | GPT-4o for complex fields (money, conditional), GPT-4o-mini for simple (date, name) | 3-5x cost reduction with minimal quality loss on easy fields |
| **Agentic retry** | If first extraction returns null, rephrase query and re-retrieve up to 3x | Handles fields that need non-obvious chunk retrieval patterns |
| **NL index lookup** | Cross-document natural language index for definitions and references | Resolves "as defined in the Master Agreement" by looking it up across docs |
| **Reviewer agent** | Post-extraction LLM that checks: is this value plausible? Is it from the text? | Catches hallucinations before they reach the database |

**Strengths vs Our Approach**:
- **Targeted retrieval**: Only processes chunks relevant to each field (vs our linear scan)
- **Built-in hallucination guard**: Reviewer agent catches bad values before storage
- **Cost-efficient**: Tiered LLM routing, Cochran sampling minimize total LLM calls
- **Cross-document context**: NL index enables reference resolution across docs

**Weaknesses vs Our Approach**:
- **Requires pre-existing field catalog**: Cannot discover new fields automatically
- **Per-field retrieval is N×K operations**: For N fields × K chunks retrieval calls
- **Reviewer agent adds latency**: Extra LLM call per extraction
- **No clustering insight**: Doesn't leverage the structure that clause-type clustering provides

**Verdict**: Complementary. RAG-based extraction is best when you **know what fields to extract**. Our approach is best when you **don't know what fields exist yet**. The ideal pipeline discovers fields first (our approach), then extracts values using RAG retrieval (their approach).

---

### 4.3 Friend's Approach B: Sequential Chunk Extraction

```
Document → Split into 4K-token chunks (optimized for extraction, not retrieval)
    ↓
For each chunk sequentially:
    ├── Carryforward context: what we've extracted so far from prior chunks
    ├── Extractor LLM: extract fields + reasoning tools (field-specific logic)
    ├── Output: extracted values + confidence + reasoning trace
    ↓
After all chunks:
    ├── Judge LLM: compile extractions across chunks
    ├── Resolve conflicts: chunk 3 says "30 days", chunk 7 says "60 days"
    ├── Merge overlapping values
    └── Final extraction record per document
```

**Key Design Decisions**:

| Feature | Description | Why It Matters |
|---|---|---|
| **4K-token chunks** | Chunks sized for extraction context, not retrieval relevance | Ensures each chunk is large enough for the LLM to understand context, but small enough to avoid context rot |
| **Carryforward context** | Each chunk extraction sees a summary of what was found in prior chunks | Prevents re-extracting already-found values; helps resolve references ("the fee described above") |
| **Field reasoning tools** | Extractor LLM has access to field-specific validation functions | E.g., money fields get currency validation, dates get format normalization |
| **Judge LLM compilation** | A separate "judge" LLM reviews all chunk-level extractions for one document and produces the final record | Resolves conflicts (different values for same field), merges partial values, flags low-confidence extractions |
| **Small model works** | Sequential processing with carryforward context mitigates context rot, so a smaller/faster model can handle extraction | Cost savings: GPT-4o-mini or Mistral-small can handle extraction when context is well-managed |

**Strengths vs Our Approach**:
- **Document-level coherence**: Carryforward context means later chunks benefit from earlier chunks
- **Conflict resolution**: Judge LLM explicitly handles value disagreements across chunks
- **Context rot mitigation**: Sequential processing keeps context window focused (no 50K-token dumps)
- **Cheaper to run**: Small models work because each call has well-scoped context
- **Reasoning transparency**: Field tools + reasoning trace make extraction auditable

**Weaknesses vs Our Approach**:
- **Sequential = slow**: Must process chunks in order (no parallelism within a document)
- **Requires pre-existing field catalog**: Like RAG, needs to know what to extract
- **No cross-document insights**: Processes each document independently
- **4K chunks may split critical clauses**: A termination clause split across two chunks loses context
- **Judge LLM bottleneck**: Reviewing all extractions for a large document is itself expensive

**Verdict**: Best approach for **per-document extraction quality**. The carryforward + judge pattern directly addresses our Gap 3 (conflict resolution) and Gap 4 (document-level context). However, it can't discover new fields.

---

## 5. Proposed Rearchitecture: Hybrid Pipeline

```
Phase 1: Cluster-First Discovery (KEEP — our existing approach)
    Documents → Chunk → Embed → Cluster → Discover Fields
    
    Changes:
    ├── Add extractability constraint to signatures
    ├── Add trial-extraction validation step
    └── Add cross-cluster field normalization

Phase 2: RAG-Optimized Retrieval (NEW — adopt from Approach A)
    For each discovered field:
    ├── Hybrid retrieval (embedding + trigram) to find top-K chunks
    └── Cochran sampling for statistical confidence
    
    Changes:
    ├── Replace linear chunk scanning
    └── Add tiered LLM routing (complex vs simple fields)

Phase 3: Sequential Extraction with Carryforward (NEW — adopt from Approach B)
    Per document, process retrieved chunks sequentially:
    ├── Carryforward context from prior chunks
    ├── Extract with field-specific reasoning
    └── Validate against source text inline
    
    Changes:
    ├── Replace one-shot extraction_from_chunk
    └── Add field reasoning tools

Phase 4: Judge Compilation + Review (NEW — adopt from both)
    ├── Judge LLM resolves conflicts across chunks per document
    ├── Reviewer agent validates grounding (from Approach A)
    └── Write final records

Phase 5: Feedback Loop (NEW)
    ├── Extraction grounding scores → retrain discovery prompts via MIPROv2
    └── Low-fill-rate fields get pruned or refined
```

### What to Build First (Priority Order)

| Priority | Change | Impact | Effort |
|---|---|---|---|
| **P0** | Fix GroundedDiscoverySig to require extractable fields | Fixes Franchise-type disasters | 1 hour |
| **P0** | Add span_text validation to overlap merge | Prevents meta-field leakage | 1 hour |
| **P1** | Trial extraction validation after discovery | Catches bad fields before full extraction | 4 hours |
| **P1** | Carryforward context in extraction loop | Fixes Gap 4 (document-level context) | 8 hours |
| **P2** | Judge LLM compilation per document | Fixes Gap 3 (conflict resolution) | 8 hours |
| **P2** | Hybrid retrieval per field | Fixes Gap 5 (retrieval optimization) | 16 hours |
| **P3** | Cross-cluster field normalization | Fixes Gap 2 (dedup across clusters) | 8 hours |
| **P3** | MIPROv2 training loop from extraction scores | Fixes Gap 6 (feedback loop) | 16 hours |
| **P3** | Tiered LLM routing | Cost optimization | 4 hours |

---

## 6. Method-by-Method Scorecard

| Capability | DSPy v2 | OG rlm_v2 | OG dspy_rlm | OG std_gpt | RAG (A) | Sequential (B) | Hybrid (Proposed) |
|---|---|---|---|---|---|---|---|
| **Field discovery** | ✅ Novel | ✅ Novel | ✅ Novel | ✅ Novel | ❌ Pre-defined | ❌ Pre-defined | ✅ Novel |
| **Grounding quality** | ⚠️ 73.6% | ⚠️ 78.7% | ✅ 97.4% | ✅ 89.7% | ✅ High (reviewer) | ✅ High (judge) | ✅ High (all three) |
| **Field count** | ✅ 15.0 | ✅ 15.0 | ⚠️ 8.6 | ⚠️ 12.0 | — | — | ✅ 15+ |
| **Fill rate** | ⚠️ 0.31 | ⚠️ 0.32 | ⚠️ 0.29 | ⚠️ 0.31 | ✅ High (targeted) | ✅ High (sequential) | ✅ High (retrieval) |
| **Cross-doc context** | ❌ | ❌ | ❌ | ❌ | ✅ NL index | ❌ | ✅ NL index |
| **Document context** | ❌ | ❌ | ❌ | ❌ | ⚠️ Partial | ✅ Carryforward | ✅ Carryforward |
| **Conflict resolution** | ❌ | ❌ | ❌ | ❌ | ⚠️ Reviewer | ✅ Judge LLM | ✅ Judge LLM |
| **Cost efficiency** | ⚠️ | ❌ Slowest | ✅ Fast | ✅ Fast | ✅ Tiered | ✅ Small model | ✅ Tiered |
| **Prompt optimization** | ✅ MIPROv2 | ❌ Manual | ❌ | ❌ | ❌ | ❌ | ✅ MIPROv2 |
| **Auditability** | ⚠️ CoT trace | ❌ Opaque | ⚠️ RLM trace | ❌ Opaque | ✅ Reviewer trace | ✅ Reasoning trace | ✅ Full trace |

---

## 7. Conclusion

**The fundamental insight**: Our pipeline solves a problem the other two don't — **field discovery**. But our extraction is naive. Their pipelines solve extraction brilliantly but assume you already know what to extract.

**The winning architecture** chains them:
1. **Our cluster-first discovery** to define what fields exist
2. **RAG retrieval** to find which chunks contain each field
3. **Sequential extraction with carryforward** to extract with document context
4. **Judge compilation** to resolve conflicts and validate

**Immediate fixes** (P0): Constraining DSPy signatures to require extractable span_text and filtering meta-fields from overlap merge would immediately bring DSPy v2 hallucination from 26.4% → ~15% (matching OG rlm_v2 excluding the Franchise outlier).
